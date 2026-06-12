// Copyright Epic Games, Inc. All Rights Reserved.

#include "ShaderParameterMacros.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// CVars
static TAutoConsoleVariable<bool> CVarNaniteCurveDebug(
	TEXT("r.Nanite.Curve.TiledRasterization.Debug"),
	false,
	TEXT("Enable debug rendering for curve rendering (default: off)."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarNaniteCurveTileCapacity(
	TEXT("r.Nanite.Curve.TiledRasterization.TileCapacity"),
	128,
	TEXT("Tile's max segments capacity."),
	ECVF_ReadOnly | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarNaniteCurveTiledRasterization(
	TEXT("r.Nanite.Curve.TiledRasterization"),
	0,
	TEXT("Enable tiled rasterization for curve primitive for primary view (default: off).\n * 0: Disabled\n * 1: Enabled for primary view\n * 2: Enabled for primary and shadow views "),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarNaniteCurveTiledRasterization_UseClusterBounds(
	TEXT("r.Nanite.Curve.TiledRasterization.UseClusterBound"),
	1,
	TEXT("Use cluster bounds for allocating tile count instead of using curves rasterization"),
	ECVF_RenderThreadSafe
);

///////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers

static bool IsNaniteCurveDebugEnabled()
{
	return CVarNaniteCurveDebug.GetValueOnRenderThread();
}

static uint32 GetNaniteCurveTileCapacity()
{
	return FMath::Max(1, CVarNaniteCurveTileCapacity.GetValueOnAnyThread());
}

static uint32 GetNaniteCurveTileWidth()
{
	return 8u;
}

static void SetNaniteCurveKernelConstants(FShaderCompilerEnvironment& OutEnvironment)
{
	OutEnvironment.SetDefine(TEXT("TILE_CAPACITY"), GetNaniteCurveTileCapacity());
	OutEnvironment.SetDefine(TEXT("TILE_WIDTH"), GetNaniteCurveTileWidth());
}

static void SetNaniteCurveKernelGroupSize(FShaderCompilerEnvironment& OutEnvironment, int32 GroupSize)
{
	if (GroupSize < 0)// negative here means that group size is set by permutation
		return;

	OutEnvironment.SetDefine(TEXT("GROUP_SIZE"), GroupSize);
}

static void SetNaniteCurveKernelGroupSize(FShaderCompilerEnvironment& OutEnvironment, FIntPoint GroupSize)
{
	OutEnvironment.SetDefine(TEXT("GROUP_SIZE_X"), GroupSize.X);
	OutEnvironment.SetDefine(TEXT("GROUP_SIZE_Y"), GroupSize.Y);
}

namespace Nanite
{

static bool SupportsCurveTiledRasterization(EShaderPlatform InShaderPlatform)
{
	return NaniteCurvesSupported() 
		&& FDataDrivenShaderPlatformInfo::GetSupportsUInt64ImageAtomics(InShaderPlatform) 
		// For now, disable support for certain platforms, as the encoding & storing routines used by the curve tiled rasterization caused shader compilation issue.
		&& !FDataDrivenShaderPlatformInfo::GetIsLanguageNintendo(InShaderPlatform)
		&& !IsVulkanPlatform(InShaderPlatform) 
		&& !IsMetalPlatform(InShaderPlatform);
}

static bool UsesCurveTiledRasterization(bool bVirtualShadowMapView)
{
	const uint32 TileRasterMode = FMath::Clamp(CVarNaniteCurveTiledRasterization.GetValueOnAnyThread(), 0, 2);
	return SupportsCurveTiledRasterization(GMaxRHIShaderPlatform) && (bVirtualShadowMapView ? TileRasterMode == 2 : TileRasterMode >= 1);
}

static void CreateCurveBinningResources(FRDGBuilder& GraphBuilder, const FIntPoint& ViewResolution, uint32 InFrameIndex, bool bIsShadowView, Nanite::FCurveResources& Out)
{
	const uint32 TileWidth = GetNaniteCurveTileWidth();
	const uint32 TileCapacity = GetNaniteCurveTileCapacity();
	const FIntPoint TileCountXY = FMath::DivideAndRoundUp(ViewResolution, FIntPoint(TileWidth));

	// per-frame constants
	Out.FrameIndex = InFrameIndex;

	// per-view constants
	Out.NumTilesX = TileCountXY.X;
	Out.NumTilesY = TileCountXY.Y;
	Out.NumTilesA = TileCountXY.X * TileCountXY.Y;
	Out.ViewSizeX = ViewResolution.X;
	Out.ViewSizeY = ViewResolution.Y;

	// per-view transient buffers
	const uint32 MaxShadowViewSegmentCount = TileCapacity * FMath::DivideAndRoundUp(4096u, TileWidth) * FMath::DivideAndRoundUp(2048u, TileWidth); // ~16M Segments: capacity for 4k x 2k VSM physical pool, with all pages/tiles occupied.
	Out.TileSegmentMaxCount = bIsShadowView ? FMath::Min(Out.NumTilesA * TileCapacity, MaxShadowViewSegmentCount) : Out.NumTilesA * TileCapacity;
	Out.TileCount = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), Out.NumTilesA), TEXT("Nanite.Curve.TileCount"));
	Out.TileOffset = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), Out.NumTilesA), TEXT("Nanite.Curve.TileOffset"));
	Out.TileMinMaxZ = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 2), TEXT("Nanite.Curve.TileMinMaxZ"));
	Out.TileSegment = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32) * 4, Out.TileSegmentMaxCount), TEXT("Nanite.Curve.TileSegment"));

	// per-view transient textures
	Out.TileHeat = GraphBuilder.CreateTexture(FRDGTextureDesc::Create2D(TileCountXY, EPixelFormat::PF_R32_FLOAT, FClearValueBinding::None, ETextureCreateFlags::ShaderResource | ETextureCreateFlags::UAV), TEXT("Nanite.Curve.TileHeat"));
	Out.OutCoverage = GraphBuilder.CreateTexture(FRDGTextureDesc::Create2D(ViewResolution, EPixelFormat::PF_R32_FLOAT, FClearValueBinding::None, ETextureCreateFlags::ShaderResource | ETextureCreateFlags::UAV), TEXT("Nanite.Curve.OutCoverage"));
	Out.OutVisibility = GraphBuilder.CreateTexture(FRDGTextureDesc::Create2D(ViewResolution, EPixelFormat::PF_R32_UINT, FClearValueBinding::None, ETextureCreateFlags::ShaderResource | ETextureCreateFlags::UAV), TEXT("Nanite.Curve.OutVisibility"));
	Out.OutVisibilityW = GraphBuilder.CreateTexture(FRDGTextureDesc::Create2D(ViewResolution, EPixelFormat::PF_R32_UINT, FClearValueBinding::None, ETextureCreateFlags::ShaderResource | ETextureCreateFlags::UAV), TEXT("Nanite.Curve.OutVisibilityW"));

	// Active tiles
	Out.ActiveTileCount = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 1), TEXT("Nanite.Curve.ActiveTileCount"));
	Out.ActiveTile = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), Out.NumTilesA), TEXT("Nanite.Curve.ActiveTile"));
	Out.ActiveTileIndirectArgs = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc<FRHIDispatchIndirectParameters>(1u), TEXT("Nanite.Curve.ActiveTileIndirectArgs"));

	// GroupIndex is filled during the AsyncTask
	//Out.GroupIndex = ...
}

}// namespace Nanite

///////////////////////////////////////////////////////////////////////////////////////////////////
// Shader parameters

BEGIN_SHADER_PARAMETER_STRUCT(FNaniteCurveCommonParameters, )

	SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)

	//--- data per-view
	SHADER_PARAMETER(uint32, FrameIndex)
	SHADER_PARAMETER(FIntVector, MaxDispatchThreadGroupsPerDimension)

	SHADER_PARAMETER(FIntPoint, RasterViewSize)
	SHADER_PARAMETER(uint32, NumTilesX)
	SHADER_PARAMETER(uint32, NumTilesY)
	SHADER_PARAMETER(uint32, NumTilesA)

	SHADER_PARAMETER(uint32, TileSegmentMaxCount)
	SHADER_PARAMETER(uint32, TileStride)

	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, TileCount)
	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, TileOffset)
	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, TileMinMaxZ)
	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint4>, TileSegment)

	SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWTileCount)
	SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWTileOffset)
	SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWTileMinMaxZ)
	SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint4>, RWTileSegment)

	// Indirect Args for active tiles
	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, ActiveTile)
	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, ActiveTileCount)
	SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWActiveTile)
	SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWActiveTileCount)
	SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWActiveTileIndirectArgs)
	RDG_BUFFER_ACCESS(ActiveTileIndirectArgs, ERHIAccess::IndirectArgs)

	SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float>, TileHeat)
	SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float>, OutCoverage)
	SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<uint>, OutVisibility)
	SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<uint>, OutVisibilityW)

	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWTileHeat)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWOutCoverage)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWOutVisibility)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWOutVisibilityW)
	//--- data per-view end

END_SHADER_PARAMETER_STRUCT()

BEGIN_SHADER_PARAMETER_STRUCT(FNaniteCurveClusterParameters, )
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER( FSceneUniformParameters, Scene )
	SHADER_PARAMETER_RDG_BUFFER_SRV( ByteAddressBuffer,		AssemblyTransforms )
	SHADER_PARAMETER_RDG_BUFFER_SRV( StructuredBuffer< FPackedView >,	InViews )
	SHADER_PARAMETER_RDG_BUFFER_SRV( StructuredBuffer<FNaniteRasterBinMeta>,	RasterBinMeta )
	SHADER_PARAMETER_RDG_BUFFER_SRV (StructuredBuffer<FNaniteRasterGroupMeta>,	RasterGroupMeta)
	SHADER_PARAMETER_RDG_BUFFER_SRV( StructuredBuffer<uint>,			RasterBinData )
	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FUintVector2>,	InTotalPrevDrawClusters)
	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FUintVector2>,	InClusterCountSWHW)
	SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>,					InClusterOffsetSWHW)
	SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, VisibleClustersSWHW)
	SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, ClusterPageData)
	SHADER_PARAMETER(FIntVector4, PageConstants)
	SHADER_PARAMETER(uint32, RenderFlags)
	SHADER_PARAMETER(uint32, MaxVisibleClusters)
	SHADER_PARAMETER(uint32, MinSupportedWaveSize)
	SHADER_PARAMETER(uint32, MaxVisiblePatches)
	SHADER_PARAMETER(uint32, MaxClusterIndirections)
	SHADER_PARAMETER(uint32, CurveSubdivisionMaxSegments)
	SHADER_PARAMETER(float,  CurveSubdivisionThreshold)
	RDG_BUFFER_ACCESS(IndirectArgs, ERHIAccess::IndirectArgs)
END_SHADER_PARAMETER_STRUCT()

BEGIN_SHADER_PARAMETER_STRUCT(FNaniteCurveBinningParameters, )
	SHADER_PARAMETER_STRUCT_INCLUDE(FNaniteCurveCommonParameters, CommonParameters)
	SHADER_PARAMETER_STRUCT_INCLUDE(FNaniteCurveClusterParameters, ClusterParameters)
	SHADER_PARAMETER_STRUCT_INCLUDE(FVirtualTargetParameters, VirtualShadowMap)
	SHADER_PARAMETER(FUintVector4, PassData)
END_SHADER_PARAMETER_STRUCT()

BEGIN_SHADER_PARAMETER_STRUCT(FNaniteCurveResolveParameters, )
	SHADER_PARAMETER_STRUCT_INCLUDE(FNaniteCurveCommonParameters, CommonParameters)
	SHADER_PARAMETER_STRUCT_INCLUDE(FRasterParameters, RasterParameters)
END_SHADER_PARAMETER_STRUCT()

BEGIN_SHADER_PARAMETER_STRUCT(FNaniteCurveDebugParameters, )
	SHADER_PARAMETER_STRUCT_INCLUDE(FNaniteCurveCommonParameters, CommonParameters)
	SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrintParameters)
END_SHADER_PARAMETER_STRUCT()

///////////////////////////////////////////////////////////////////////////////////////////////////
// Shaders

#define NAMEOF(ShaderClass) #ShaderClass

#define SHADER_STRUCT_SIG(ShaderClass)												class FNanite##ShaderClass : public FNaniteGlobalShader
#define SHADER_STRUCT_INC(ShaderClass, ParameterStruct, GroupSize)					\
	DECLARE_GLOBAL_SHADER(FNanite##ShaderClass);									\
	SHADER_USE_PARAMETER_STRUCT(FNanite##ShaderClass, FNaniteGlobalShader);			\
	using FParameters = ParameterStruct;											\
	static auto GetGroupSize() { return GroupSize; }
#define SHADER_MODIFY_SIG(Parameters, OutEnvironment)								static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
#define SHADER_MODIFY_INC(Parameters, OutEnvironment)								\
	FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);	\
	SetNaniteCurveKernelConstants(OutEnvironment);									\
	SetNaniteCurveKernelGroupSize(OutEnvironment, GetGroupSize());
#define SHADER_CONDITION_SIG(Parameters)											static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
#define SHADER_CONDITION_INC(Parameters)											\
	(FNaniteGlobalShader::ShouldCompilePermutation(Parameters) &&					\
	 Nanite::SupportsCurveTiledRasterization(Parameters.Platform))
#define SHADER_PRECACHED_SIG(Parameters)											static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)

#define IMPLEMENT_GLOBAL_SHADER_STRUCT(ShaderClass, ParameterStruct, SourceFilename, Frequency, GroupSize, ...)\
	SHADER_STRUCT_SIG(ShaderClass) { SHADER_STRUCT_INC(ShaderClass, ParameterStruct, __VA_OPT__(-) GroupSize)\
		__VA_OPT__(class FGroupSizeDim : SHADER_PERMUTATION_SPARSE_INT("GROUP_SIZE", GroupSize, __VA_ARGS__);)\
		using FPermutationDomain = TShaderPermutationDomain<__VA_OPT__(FGroupSizeDim)>;			\
		SHADER_CONDITION_SIG(P) { return SHADER_CONDITION_INC(P); }								\
		SHADER_MODIFY_SIG(P, O) { SHADER_MODIFY_INC(P, O) }										\
	};																							\
	IMPLEMENT_GLOBAL_SHADER(FNanite##ShaderClass, SourceFilename, NAMEOF(ShaderClass), Frequency)

#define IMPLEMENT_GLOBAL_SHADER_STRUCT_EX2(ShaderClass, ParameterStruct, SourceFilename, Frequency, GroupSize)\
	SHADER_STRUCT_SIG(ShaderClass) { SHADER_STRUCT_INC(ShaderClass, ParameterStruct, GroupSize)	\
		class FMultiViewDim : SHADER_PERMUTATION_BOOL("NANITE_MULTI_VIEW");						\
		class FVirtualTextureTargetDim : SHADER_PERMUTATION_BOOL("VIRTUAL_TEXTURE_TARGET");		\
		using FPermutationDomain = TShaderPermutationDomain<FVirtualTextureTargetDim, FMultiViewDim>;\
		SHADER_CONDITION_SIG(P) { return SHADER_CONDITION_INC(P); }								\
		SHADER_MODIFY_SIG(P, O) { SHADER_MODIFY_INC(P, O) }										\
	};																							\
	IMPLEMENT_GLOBAL_SHADER(FNanite##ShaderClass, SourceFilename, NAMEOF(ShaderClass), Frequency)

#define IMPLEMENT_GLOBAL_SHADER_STRUCT_SP(ShaderClass, ParameterStruct, SourceFilename, Frequency, GroupSize)\
	SHADER_STRUCT_SIG(ShaderClass) { SHADER_STRUCT_INC(ShaderClass, ParameterStruct, GroupSize)	\
		using FPermutationDomain = TShaderPermutationDomain<>;									\
		SHADER_PRECACHED_SIG(P) { return EShaderPermutationPrecacheRequest::NotPrecached; }		\
		SHADER_CONDITION_SIG(P) { return SHADER_CONDITION_INC(P); }								\
		SHADER_MODIFY_SIG(P, O) { SHADER_MODIFY_INC(P, O)										\
			ShaderPrint::ModifyCompilationEnvironment(P, O);									\
		}																						\
	};																							\
	IMPLEMENT_GLOBAL_SHADER(FNanite##ShaderClass, SourceFilename, NAMEOF(ShaderClass), Frequency)

IMPLEMENT_GLOBAL_SHADER_STRUCT(CurveTileClear,			FNaniteCurveCommonParameters,	"/Engine/Private/Nanite/NaniteCurve.usf", SF_Compute, 64);
IMPLEMENT_GLOBAL_SHADER_STRUCT(CurveTileCountCap,		FNaniteCurveCommonParameters,	"/Engine/Private/Nanite/NaniteCurve.usf", SF_Compute, 64);
IMPLEMENT_GLOBAL_SHADER_STRUCT(CurveTileIndirectArgs,	FNaniteCurveCommonParameters,	"/Engine/Private/Nanite/NaniteCurve.usf", SF_Compute, 1);
IMPLEMENT_GLOBAL_SHADER_STRUCT(CurveTileOffsetScan,		FNaniteCurveCommonParameters,	"/Engine/Private/Nanite/NaniteCurve.usf", SF_Compute, 16, 64, 256, 1024);
IMPLEMENT_GLOBAL_SHADER_STRUCT(CurveTileOffsetMerge,	FNaniteCurveCommonParameters,	"/Engine/Private/Nanite/NaniteCurve.usf", SF_Compute, 16, 64, 256, 1024);

IMPLEMENT_GLOBAL_SHADER_STRUCT(CurveTileResolve,		FNaniteCurveResolveParameters,	"/Engine/Private/Nanite/NaniteCurve.usf", SF_Compute, FIntPoint(GetNaniteCurveTileWidth()));
IMPLEMENT_GLOBAL_SHADER_STRUCT_EX2(CurveTileTranslate,	FNaniteCurveResolveParameters,	"/Engine/Private/Nanite/NaniteCurve.usf", SF_Compute, FIntPoint(GetNaniteCurveTileWidth()));

IMPLEMENT_GLOBAL_SHADER_STRUCT_SP(CurveTileDebug,		FNaniteCurveDebugParameters,	"/Engine/Private/Nanite/NaniteCurve.usf", SF_Compute, FIntPoint(GetNaniteCurveTileWidth()));

#undef IMPLEMENT_GLOBAL_SHADER_STRUCT
#undef IMPLEMENT_GLOBAL_SHADER_STRUCT_EX2
#undef IMPLEMENT_GLOBAL_SHADER_STRUCT_SP

class FNaniteCurveTileCount : public FNaniteGlobalShader
{
	DECLARE_GLOBAL_SHADER(FNaniteCurveTileCount);
	SHADER_USE_PARAMETER_STRUCT(FNaniteCurveTileCount, FNaniteGlobalShader);

	class FVirtualTextureTargetDim : SHADER_PERMUTATION_BOOL("VIRTUAL_TEXTURE_TARGET");
	class FMultiViewDim : SHADER_PERMUTATION_BOOL("NANITE_MULTI_VIEW");
	class FUseClusterBounds : SHADER_PERMUTATION_BOOL("USE_CLUSTER_BOUNDS");
	using FPermutationDomain = TShaderPermutationDomain<FVirtualTextureTargetDim, FMultiViewDim, FUseClusterBounds>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FNaniteCurveClusterParameters, ClusterParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FNaniteCurveCommonParameters, CommonParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE( FVirtualTargetParameters, VirtualShadowMap )
		SHADER_PARAMETER(FUintVector4, PassData)
	END_SHADER_PARAMETER_STRUCT()

	static uint32 GetGroupSize() { return 64; }

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return 
			FNaniteGlobalShader::ShouldCompilePermutation(Parameters) && 
			Nanite::SupportsCurveTiledRasterization(Parameters.Platform); 
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		OutEnvironment.SetDefine(TEXT("GROUP_SIZE"), GetGroupSize());
		OutEnvironment.SetDefine(TEXT("NANITE_CURVES"), 1);
		SetNaniteCurveKernelConstants(OutEnvironment);
		OutEnvironment.CompilerFlags.Add(CFLAG_RootConstants);
		OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
		FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	}
};
class FNaniteCurveTileWrite : public FNaniteGlobalShader
{
	DECLARE_GLOBAL_SHADER(FNaniteCurveTileWrite);
	SHADER_USE_PARAMETER_STRUCT(FNaniteCurveTileWrite, FNaniteGlobalShader);

	class FVirtualTextureTargetDim : SHADER_PERMUTATION_BOOL("VIRTUAL_TEXTURE_TARGET");
	class FMultiViewDim : SHADER_PERMUTATION_BOOL("NANITE_MULTI_VIEW");
	using FPermutationDomain = TShaderPermutationDomain<FVirtualTextureTargetDim, FMultiViewDim>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FNaniteCurveClusterParameters, ClusterParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FNaniteCurveCommonParameters, CommonParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE( FVirtualTargetParameters, VirtualShadowMap )
		SHADER_PARAMETER(FUintVector4, PassData)
	END_SHADER_PARAMETER_STRUCT()

	static uint32 GetGroupSize() { return 64; }

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return 
			FNaniteGlobalShader::ShouldCompilePermutation(Parameters) && 
			Nanite::SupportsCurveTiledRasterization(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		OutEnvironment.SetDefine(TEXT("GROUP_SIZE"), GetGroupSize());
		OutEnvironment.SetDefine(TEXT("NANITE_CURVES"), 1);
		SetNaniteCurveKernelConstants(OutEnvironment);
		OutEnvironment.CompilerFlags.Add(CFLAG_RootConstants);
		OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
		FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	}
};
IMPLEMENT_GLOBAL_SHADER(FNaniteCurveTileCount, "/Engine/Private/Nanite/NaniteRasterizer.usf", "CurveTileCount", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FNaniteCurveTileWrite, "/Engine/Private/Nanite/NaniteRasterizer.usf", "CurveTileWrite", SF_Compute);

///////////////////////////////////////////////////////////////////////////////////////////////////

static void SetCurveCommonParameters(FRDGBuilder& GraphBuilder, const FViewInfo& View, const Nanite::FCurveResources& Resources, FNaniteCurveCommonParameters& Out)
{
	Out.View = View.ViewUniformBuffer;
	Out.RasterViewSize = Resources.OutCoverage->Desc.Extent;

	Out.FrameIndex = Resources.FrameIndex;
	Out.MaxDispatchThreadGroupsPerDimension = GRHIMaxDispatchThreadGroupsPerDimension;

	Out.NumTilesX = Resources.NumTilesX;
	Out.NumTilesY = Resources.NumTilesY;
	Out.NumTilesA = Resources.NumTilesA;
	Out.TileSegmentMaxCount = Resources.TileSegmentMaxCount;
	
	Out.TileStride = 1;

	Out.TileCount = GraphBuilder.CreateSRV(Resources.TileCount);
	Out.TileOffset = GraphBuilder.CreateSRV(Resources.TileOffset);
	Out.TileMinMaxZ = GraphBuilder.CreateSRV(Resources.TileMinMaxZ);
	Out.TileSegment = GraphBuilder.CreateSRV(Resources.TileSegment);

	Out.RWTileCount = GraphBuilder.CreateUAV(Resources.TileCount);
	Out.RWTileOffset = GraphBuilder.CreateUAV(Resources.TileOffset);
	Out.RWTileMinMaxZ = GraphBuilder.CreateUAV(Resources.TileMinMaxZ);
	Out.RWTileSegment = GraphBuilder.CreateUAV(Resources.TileSegment);

	Out.TileHeat = GraphBuilder.CreateSRV(Resources.TileHeat);
	Out.OutCoverage = GraphBuilder.CreateSRV(Resources.OutCoverage);
	Out.OutVisibility = GraphBuilder.CreateSRV(Resources.OutVisibility);
	Out.OutVisibilityW = GraphBuilder.CreateSRV(Resources.OutVisibilityW);

	Out.RWTileHeat = GraphBuilder.CreateUAV(Resources.TileHeat);
	Out.RWOutCoverage = GraphBuilder.CreateUAV(Resources.OutCoverage);
	Out.RWOutVisibility = GraphBuilder.CreateUAV(Resources.OutVisibility);
	Out.RWOutVisibilityW = GraphBuilder.CreateUAV(Resources.OutVisibilityW);

	Out.ActiveTileCount = GraphBuilder.CreateSRV(Resources.ActiveTileCount);
	Out.ActiveTile = GraphBuilder.CreateSRV(Resources.ActiveTile);
	Out.RWActiveTileCount = GraphBuilder.CreateUAV(Resources.ActiveTileCount);
	Out.RWActiveTile = GraphBuilder.CreateUAV(Resources.ActiveTile);

	Out.ActiveTileIndirectArgs = Resources.ActiveTileIndirectArgs;
	Out.RWActiveTileIndirectArgs = GraphBuilder.CreateUAV(Resources.ActiveTileIndirectArgs, PF_R32_UINT);
}

namespace Nanite
{

void FRenderer::SetCurveClusterParameters(
	const FBinningData& BinningData,
	FRDGBufferRef ClusterOffsetSWHW,
	FRDGBufferRef IndirectArgs,
	FNaniteCurveClusterParameters& Out)
{
	Out.Scene						= SceneUniformBuffer;
	Out.VisibleClustersSWHW			= GraphBuilder.CreateSRV(VisibleClustersSWHW);
	Out.ClusterPageData				= GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
	Out.InClusterCountSWHW			= GraphBuilder.CreateSRV(ClusterCountSWHW);
	Out.InClusterOffsetSWHW			= GraphBuilder.CreateSRV(ClusterOffsetSWHW, PF_R32_UINT);
	Out.RasterBinMeta				= GraphBuilder.CreateSRV(BinningData.BinMetaBuffer);
	Out.RasterGroupMeta				= GraphBuilder.CreateSRV(BinningData.GroupMetaBuffer);
	Out.RasterBinData				= GraphBuilder.CreateSRV(BinningData.DataBuffer);
	Out.AssemblyTransforms			= GraphBuilder.CreateSRV(AssemblyTransformsBuffer);
	Out.IndirectArgs				= IndirectArgs;
	Out.InViews						= GraphBuilder.CreateSRV(ViewsBuffer);
	Out.PageConstants 				= PageConstants;
	Out.RenderFlags					= RenderFlags;
	Out.MaxVisibleClusters 			= Nanite::FGlobalResources::GetMaxVisibleClusters();
	Out.MinSupportedWaveSize		= GRHIMinimumWaveSize;
	Out.MaxClusterIndirections		= GetMaxClusterIndirections();
	Out.CurveSubdivisionThreshold	= Nanite::FGlobalResources::GetCurveSubdivisionThreshold();
	Out.CurveSubdivisionMaxSegments = Nanite::FGlobalResources::GetCurveSubdivisionMaxSegments();
	Out.MaxVisiblePatches			= Nanite::FGlobalResources::GetMaxVisiblePatches();

	check(Out.MaxClusterIndirections > 0);
}

template<typename TShaderClass>
static void DispatchIndirectHelper(
	FRHIComputeCommandList& RHICmdList, 
	const TShaderRef<TShaderClass>& ComputeShader,
	typename TShaderClass::FParameters& Parameters,
	FRDGBufferRef IndirectArgs,
	uint32 InGroupIndex)
{
	const uint32 IndirectOffset = GetRasterGroupIndirectOffset(InGroupIndex);
	FRHIBuffer* IndirectArgsBuffer = IndirectArgs->GetIndirectRHICallBuffer();
	FRHIComputeShader* ShaderRHI = ComputeShader.GetComputeShader();
	FComputeShaderUtils::ValidateIndirectArgsBuffer(IndirectArgsBuffer->GetSize(), IndirectOffset);
	SetComputePipelineState(RHICmdList, ShaderRHI);

	// Root constant setup
	Parameters.PassData = FUintVector4(InGroupIndex, 0u, 0u, 0u);
	if (GRHISupportsShaderRootConstants)
	{
		RHICmdList.SetComputeShaderRootConstants(Parameters.PassData);
	}

	const FShaderParametersMetadata* ParametersMetadata = TShaderClass::FParameters::FTypeInfo::GetStructMetadata();
	SetShaderParameters(RHICmdList, ComputeShader, ShaderRHI, ParametersMetadata, Parameters);
	RHICmdList.DispatchIndirectComputeShader(IndirectArgsBuffer, IndirectOffset);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ShaderRHI);	
}

void FRenderer::AddPass_CurveTiledRasterize(
	const FDispatchContext& DispatchContext,
	const FBinningData& BinningData,
	FRDGBufferRef ClusterOffsetSWHW,
	FRDGBufferRef IndirectArgs,
	ERDGPassFlags PassFlags)
{
	check(DispatchContext.CurveResources.IsValid());

	// per view resources
	const FCurveResources& Resources = DispatchContext.CurveResources;
	const FGlobalShaderMap* ShaderMap = SharedContext.ShaderMap;
	const FViewInfo& View = SceneView;

	// per view constants
	const uint32 NumTilesX = Resources.NumTilesX;
	const uint32 NumTilesY = Resources.NumTilesY;
	const uint32 NumTilesA = Resources.NumTilesA;
	const uint32 ViewSizeX = Resources.ViewSizeX;
	const uint32 ViewSizeY = Resources.ViewSizeY;
	const bool bUseClusterBounds = CVarNaniteCurveTiledRasterization_UseClusterBounds.GetValueOnRenderThread() > 0;

	// default parameters
	FNaniteCurveCommonParameters DefaultCommonParams;
	{
		SetCurveCommonParameters(GraphBuilder, View, Resources, DefaultCommonParams);
	}
	FNaniteCurveBinningParameters DefaultBinningParams;
	{
		DefaultBinningParams.CommonParameters = DefaultCommonParams;
		SetCurveClusterParameters(BinningData, ClusterOffsetSWHW, IndirectArgs, DefaultBinningParams.ClusterParameters);
	}
	FNaniteCurveResolveParameters DefaultResolveParams;
	{
		DefaultResolveParams.CommonParameters = DefaultCommonParams;
		DefaultResolveParams.RasterParameters = RasterContext.Parameters;
	}

	// binning
	//	i.		clear tiles
	//	ii.		count segments over tiles
	//	iii.	calc tile heat
	//	iv.		calc tile offsets
	//	v.		write segments into tiles
	{
		RDG_EVENT_SCOPE(GraphBuilder, "Nanite::CurveBinning");

		// i. clear tiles
		{
			auto Params = GraphBuilder.AllocParameters(&DefaultCommonParams);
			auto Kernel = ShaderMap->GetShader<FNaniteCurveTileClear>();
			{
				FComputeShaderUtils::AddPass(
					GraphBuilder,
					RDG_EVENT_NAME("CurveTileClear"),
					PassFlags,
					Kernel,
					Params,
					FComputeShaderUtils::GetGroupCount(NumTilesA, FNaniteCurveTileClear::GetGroupSize())
				);
			}
		}

		// ii. count segments over tiles
		{
			auto* Params = GraphBuilder.AllocParameters<FNaniteCurveTileCount::FParameters>();
			SetCurveClusterParameters(BinningData, ClusterOffsetSWHW, IndirectArgs, Params->ClusterParameters);
			Params->CommonParameters = DefaultCommonParams;
			if (IsUsingVirtualShadowMap())
			{
				Params->VirtualShadowMap = VirtualTargetParameters;
			}

			FNaniteCurveTileCount::FPermutationDomain PermutationVector;
			PermutationVector.Set<FNaniteCurveTileCount::FUseClusterBounds>(bUseClusterBounds);
			PermutationVector.Set<FNaniteCurveTileCount::FMultiViewDim>(bMultiView);
			PermutationVector.Set<FNaniteCurveTileCount::FVirtualTextureTargetDim>(IsUsingVirtualShadowMap());
			const TShaderRef<FNaniteCurveTileCount>& Kernel = SharedContext.ShaderMap->GetShader<FNaniteCurveTileCount>(PermutationVector);
			ClearUnusedGraphResources(Kernel, Params);

			GraphBuilder.AddPass(
				RDG_EVENT_NAME("CurveTileCount"),
				Params,
				PassFlags,
				[Params, &DispatchContext, Kernel](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
				{
					DispatchIndirectHelper(RHICmdList, Kernel, *Params, Params->ClusterParameters.IndirectArgs, DispatchContext.CurveResources.GroupIndex);
				}
			);
		}

		// iii. calc tile heat
		{
			{
				auto Params = GraphBuilder.AllocParameters(&DefaultCommonParams);
				auto Kernel = ShaderMap->GetShader<FNaniteCurveTileCountCap>();
				{
					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("CurveTileCountCap"),
						PassFlags,
						Kernel,
						Params,
						FComputeShaderUtils::GetGroupCount(NumTilesA, FNaniteCurveTileCountCap::GetGroupSize())
					);
				}
			}

			{
				auto Params = GraphBuilder.AllocParameters(&DefaultCommonParams);
				auto Kernel = ShaderMap->GetShader<FNaniteCurveTileIndirectArgs>();
				{
					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("CurveTileIndirectArgs"),
						PassFlags,
						Kernel,
						Params,
						FIntVector(1,1,1)
					);
				}
			}
		}

		// iv. calc tile offsets
		{
			// minimum wave size decides group size
			//TODO: maybe pick according to max wave size and supply [WaveSize()] in hlsl
			int GroupSize = -1;
			{
				switch (GRHIMinimumWaveSize)
				{
				case 4: GroupSize = 16; break;
				case 8: GroupSize = 64; break;
				case 16: GroupSize = 256; break;
				case 32:
				case 64:
				case 128: GroupSize = 1024; break;
				default:
					check(false);
				}
			}

			FNaniteCurveTileOffsetScan::FPermutationDomain PermutationVectorScan;
			FNaniteCurveTileOffsetMerge::FPermutationDomain PermutationVectorMerge;
			{
				PermutationVectorScan.Set<FNaniteCurveTileOffsetScan::FGroupSizeDim>(GroupSize);
				PermutationVectorMerge.Set<FNaniteCurveTileOffsetMerge::FGroupSizeDim>(GroupSize);
			}

			auto Params = GraphBuilder.AllocParameters(&DefaultCommonParams);
			auto KernelScan = ShaderMap->GetShader<FNaniteCurveTileOffsetScan>(PermutationVectorScan);
			auto KernelMerge = ShaderMap->GetShader<FNaniteCurveTileOffsetMerge>(PermutationVectorMerge);
			{
				uint32& RefStride = Params->TileStride;

				// block scan
				for (RefStride = 1; RefStride < NumTilesA; RefStride *= GroupSize)
				{
					//LOG("Scan (Block) Stride %d => Tile Subset %d, Warps %d", RefTileStride, FMath::DivideAndRoundUp(NumTilesA, RefTileStride), FMath::DivideAndRoundUp(FMath::DivideAndRoundUp(NumTilesA, TileStride), WaveSize));
					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("CurveTileOffsetScan"),
						PassFlags,
						KernelScan,
						GraphBuilder.AllocParameters(Params),// copy to instantiate changing stride
						FComputeShaderUtils::GetGroupCount(FMath::DivideAndRoundUp(NumTilesA, RefStride), GroupSize)
					);
				}

				// block merge
				for (RefStride /= (GroupSize * GroupSize); RefStride >= 1; RefStride /= GroupSize)
				{
					//LOG("Merge (Block) Stride %d => Tile Subset %d, Warps %d", TileStride, FMath::DivideAndRoundUp(NumTilesA, RefTileStride), FMath::DivideAndRoundUp(FMath::DivideAndRoundUp(NumTilesA, RefTileStride), WaveSize));
					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("CurveTileOffsetMerge"),
						PassFlags,
						KernelMerge,
						GraphBuilder.AllocParameters(Params),// copy to instantiate changing stride
						FComputeShaderUtils::GetGroupCount(FMath::DivideAndRoundUp(NumTilesA, RefStride), GroupSize)
					);
				}
			}
		}

		// v. write segments into tiles
		{
			auto* Params = GraphBuilder.AllocParameters<FNaniteCurveTileWrite::FParameters>();
			SetCurveClusterParameters(BinningData, ClusterOffsetSWHW, IndirectArgs, Params->ClusterParameters);
			Params->CommonParameters = DefaultCommonParams;
			if (IsUsingVirtualShadowMap())
			{
				Params->VirtualShadowMap = VirtualTargetParameters;
			}

			FNaniteCurveTileWrite::FPermutationDomain PermutationVector;
			PermutationVector.Set<FNaniteCurveTileWrite::FMultiViewDim>(bMultiView);
			PermutationVector.Set<FNaniteCurveTileWrite::FVirtualTextureTargetDim>(IsUsingVirtualShadowMap());
			const TShaderRef<FNaniteCurveTileWrite>& Kernel = SharedContext.ShaderMap->GetShader<FNaniteCurveTileWrite>(PermutationVector);
			ClearUnusedGraphResources(Kernel, Params);

			GraphBuilder.AddPass(
				RDG_EVENT_NAME("CurveTileWrite"),
				Params,
				PassFlags,
				[Params, &DispatchContext, Kernel](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
				{
					DispatchIndirectHelper(RHICmdList, Kernel, *Params, Params->ClusterParameters.IndirectArgs, DispatchContext.CurveResources.GroupIndex);
				}
			);
		}
	}
	
	// resolve
	//	i.		resolve tiles (dedicated fine raster)
	//	ii.		translate to nanite visibility
	{
		RDG_EVENT_SCOPE(GraphBuilder, "Nanite::CurveResolve");

		// i. resolve tiles (dedicated fine raster)
		{
			auto Params = GraphBuilder.AllocParameters(&DefaultResolveParams);
			auto Kernel = ShaderMap->GetShader<FNaniteCurveTileResolve>();
			{
				FComputeShaderUtils::AddPass(
					GraphBuilder,
					RDG_EVENT_NAME("CurveTileResolve"),
					PassFlags,
					Kernel,
					Params,
					Params->CommonParameters.ActiveTileIndirectArgs, 0u
				);
			}
		}

		// ii. translate to nanite visibility
		{
			FNaniteCurveTileTranslate::FPermutationDomain PermutationVector;
			PermutationVector.Set<FNaniteCurveTileTranslate::FMultiViewDim>(bMultiView);
			PermutationVector.Set<FNaniteCurveTileTranslate::FVirtualTextureTargetDim>(IsUsingVirtualShadowMap());

			auto Params = GraphBuilder.AllocParameters(&DefaultResolveParams);
			auto Kernel = ShaderMap->GetShader<FNaniteCurveTileTranslate>(PermutationVector);
			{
				FComputeShaderUtils::AddPass(
					GraphBuilder,
					RDG_EVENT_NAME("CurveTileTranslate"),
					PassFlags,
					Kernel,
					Params,
					Params->CommonParameters.ActiveTileIndirectArgs, 0u
				);
			}
		}
	}

	// done
}

void FRenderer::AddPass_CurveDebug(const FDispatchContext& DispatchContext)
{
	if (!DispatchContext.CurveResources.IsValid())
	{
		return;
	}

	// per view resources
	const FCurveResources& Resources = DispatchContext.CurveResources;
	const FGlobalShaderMap* ShaderMap = SharedContext.ShaderMap;
	const FViewInfo& View = SceneView;

	// per view constants
	const uint32 NumTilesX = Resources.NumTilesX;
	const uint32 NumTilesY = Resources.NumTilesY;
	const uint32 NumTilesA = Resources.NumTilesA;

	ShaderPrint::SetEnabled(true);
	ShaderPrint::RequestSpaceForCharacters(4096);
	ShaderPrint::RequestSpaceForLines(NumTilesA * 4 + GetNaniteCurveTileCapacity() * 10);
	ShaderPrint::RequestSpaceForTriangles(NumTilesA * 2);

	// debug output
	{
		FNaniteCurveTileDebug::FPermutationDomain PermutationVector;
		auto Params = GraphBuilder.AllocParameters<FNaniteCurveDebugParameters>();
		auto Kernel = ShaderMap->GetShader<FNaniteCurveTileDebug>(PermutationVector);

		SetCurveCommonParameters(GraphBuilder, View, Resources, Params->CommonParameters);
		ShaderPrint::SetParameters(GraphBuilder, Params->ShaderPrintParameters);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("CurveTileDebug"),
			Kernel,
			Params,
			Params->CommonParameters.ActiveTileIndirectArgs, 0u
		);
	}
}

} // namespace Nanite
