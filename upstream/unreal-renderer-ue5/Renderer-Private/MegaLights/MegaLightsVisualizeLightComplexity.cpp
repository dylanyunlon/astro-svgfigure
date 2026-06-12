// Copyright Epic Games, Inc. All Rights Reserved.

#include "MegaLights.h"
#include "MegaLightsInternal.h"
#include "MegaLightsVisualizationData.h"
#include "PixelShaderUtils.h"
#include "HAL/IConsoleManager.h"
#include "LogRenderer.h"

static TAutoConsoleVariable<int32> CVarMegaLightsVisualizeTileClassification(
	TEXT("r.MegaLights.Visualize.TileClassification"),
	0,
	TEXT("Whether to visualize tile classification.\n")
	TEXT("0 - Disable\n")
	TEXT("1 - Visualize tiles\n")
	TEXT("2 - Visualize downsampled tiles\n")
	TEXT("3 - Visualize tiles for hair strands\n")
	TEXT("4 - Visualize downsampled tiles for hair strands\n")
	TEXT("5 - Visualize tiles for front layer translucency\n")
	TEXT("6 - Visualize downsampled tiles for front layer translucency"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVisualizeLightComplexity(
	TEXT("r.MegaLights.Visualize.LightComplexity"),
	0,
	TEXT("Whether to visualize light complexity.\n")
	TEXT("1 - opaque\n")
	TEXT("2 - hair strands\n")
	TEXT("3 - front layer translucency\n")
	TEXT("4 - volume\n")
	TEXT("5 - translucency lighting volume."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVisualizeLightComplexityMaxLightsToDisplay(
	TEXT("r.MegaLights.Visualize.LightComplexity.MaxLightsToDisplay"),
	32,
	TEXT("Maximum number of lights to display on screen."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsVisualizeLightComplexityMaxLightSamplingCost(
	TEXT("r.MegaLights.Visualize.LightComplexity.MaxLightSamplingCost"),
	32.0f,
	TEXT("Light sampling cost normalization factor. The normalized cost is used to colorize pixels in the heatmap view."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVisualizeLightComplexityFreeze(
	TEXT("r.MegaLights.Visualize.LightComplexity.Freeze"),
	0,
	TEXT("Whether to freeze the visualization."),
	ECVF_RenderThreadSafe
);

int32 GMegaLightsVisualizeLightComplexityDump = 0;
static FAutoConsoleCommand CmdMegaLightsVisualizeLightComplexityDump(
	TEXT("r.MegaLights.Visualize.LightComplexity.Dump"),
	TEXT("Dump the list of visualized lights to log."),
	FConsoleCommandDelegate::CreateLambda([]()
		{
			GMegaLightsVisualizeLightComplexityDump = 1;
		})
);

bool FMegaLightsViewContext::HasValidLightSamplingCostEstimate() const
{
	return LightSamplingCostEstimate != nullptr;
}

namespace MegaLights
{
	int32 GetDebugTileClassificationMode()
	{
		return CVarMegaLightsVisualizeTileClassification.GetValueOnRenderThread();
	}

	EMegaLightsDebugTarget GetVisualizeLightComplexityTarget(const FViewInfo& View)
	{
		const FMegaLightsVisualizationData& VisualizationData = GetMegaLightsVisualizationData();
		const FMegaLightsVisualizationData::EModeID ModeID = View.Family && View.Family->EngineShowFlags.VisualizeMegaLights ?
			VisualizationData.GetModeID(View.CurrentMegaLightsVisualizationMode) :
			FMegaLightsVisualizationData::EModeID::Invalid;

		int32 Target;
		switch (ModeID)
		{
		case FMegaLightsVisualizationData::EModeID::LightComplexity_Opaque:
			Target = 0;
			break;
		case FMegaLightsVisualizationData::EModeID::LightComplexity_HairStrands:
			Target = 1;
			break;
		case FMegaLightsVisualizationData::EModeID::LightComplexity_FrontLayerTranslucency:
			Target = 2;
			break;
		case FMegaLightsVisualizationData::EModeID::LightComplexity_Volume:
			Target = 3;
			break;
		case FMegaLightsVisualizationData::EModeID::LightComplexity_TranslucencyLightingVolume:
			Target = 4;
			break;
		default:
			Target = FMath::Clamp(CVarMegaLightsVisualizeLightComplexity.GetValueOnRenderThread(), 0, 5) - 1;
			break;
		}

		return Target >= 0 ? EMegaLightsDebugTarget(1u << Target) : EMegaLightsDebugTarget::None;
	}

	bool IsUsingLightNames(const FViewInfo& View)
	{
		return GetVisualizeLightComplexityTarget(View) != EMegaLightsDebugTarget::None;
	}

	bool IsVisualizeLightComplexityFrozen()
	{
		return CVarMegaLightsVisualizeLightComplexityFreeze.GetValueOnRenderThread() != 0;
	}

	bool ShouldShowLightSamplingCostHeatmap(const FViewInfo& View, EMegaLightsInput InputType)
	{
		EMegaLightsDebugTarget TargetMask;
		switch (InputType)
		{
		case EMegaLightsInput::GBuffer:
			TargetMask = EMegaLightsDebugTarget::GBuffer;
			break;
		case EMegaLightsInput::HairStrands:
			TargetMask = EMegaLightsDebugTarget::HairStrands;
			break;
		case EMegaLightsInput::FrontLayerTranslucency:
			TargetMask = EMegaLightsDebugTarget::FrontLayerTranslucency;
			break;
		case EMegaLightsInput::Count:
			TargetMask = EMegaLightsDebugTarget::GBuffer | EMegaLightsDebugTarget::HairStrands | EMegaLightsDebugTarget::FrontLayerTranslucency;
			break;
		default:
			checkNoEntry();
			TargetMask = EMegaLightsDebugTarget::None;
			break;
		}

		return (GetVisualizeLightComplexityTarget(View) & TargetMask) != EMegaLightsDebugTarget::None
			&& !IsVisualizeLightComplexityFrozen()
			&& CVarMegaLightsVisualizeLightComplexityMaxLightSamplingCost.GetValueOnRenderThread() > 0.0f;
	}

	bool ShouldAddVisualizeLightComplexityPostProcessingPass(const FViewInfo& View, int32 ViewIndex, const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries)
	{
		return View.Family
			&& IsEnabled(*View.Family)
			&& ShouldShowLightSamplingCostHeatmap(View, EMegaLightsInput::Count)
			&& MegaLightsFrameTemporaries
			&& MegaLightsFrameTemporaries->ViewContexts[ViewIndex].HasValidLightSamplingCostEstimate();
	}
}

namespace MegaLightsLightComplexity
{
	// Keep in-sync with MegaLightsLightComplexity.usf
	const uint32 GlobalFrozenDataDwords = 59;
	const uint32 LightFrozenDataExceptNameDwords = 8;

	// Total: 1 dword
	// 0: Latched scene depth. Zero means nothing is latched
	const uint32 GeneralPurposeStateDwords = 1;

	// Total: 1 dword
	// 0: bit0 - bIsWritten, bit1 - bInnerTLVSelected
	const uint32 GeneralPurposeFeedbackDwords = 1;
}

class FMegaLightsDebugCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMegaLightsDebugCS)
	SHADER_USE_PARAMETER_STRUCT(FMegaLightsDebugCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		RDG_BUFFER_ACCESS(IndirectArgs, ERHIAccess::IndirectArgs)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, TileAllocator)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, TileData)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, DownsampledTileAllocator)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, DownsampledTileData)
		SHADER_PARAMETER(uint32, DebugTileClassificationMode)
	END_SHADER_PARAMETER_STRUCT()

	static int32 GetGroupSize()
	{
		return 64;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return EShaderPermutationPrecacheRequest::NotPrecached;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FMegaLightsDebugCS, "/Engine/Private/MegaLights/MegaLightsDebug.usf", "MegaLightsDebugCS", SF_Compute);

class FMegaLightsSortLightsForDisplayCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMegaLightsSortLightsForDisplayCS)
	SHADER_USE_PARAMETER_STRUCT(FMegaLightsSortLightsForDisplayCS, FGlobalShader)

	class FInputType : SHADER_PERMUTATION_INT("INPUT_TYPE", int32(EMegaLightsInput::Count));
	class FIsVisualizingVolume : SHADER_PERMUTATION_BOOL("IS_VISUALIZING_VOLUME");
	class FTranslucencyLightingVolume : SHADER_PERMUTATION_BOOL("TRANSLUCENCY_LIGHTING_VOLUME");
	class FHasLightVisibilityHistory : SHADER_PERMUTATION_BOOL("HAS_LIGHT_VISIBILITY_HISTORY");
	using FPermutationDomain = TShaderPermutationDomain<FInputType, FIsVisualizingVolume, FTranslucencyLightingVolume, FHasLightVisibilityHistory>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsVolumeParameters, MegaLightsVolumeParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(StochasticLighting::FHistoryScreenParameters, HistoryScreenParameters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWGeneralPurposeState)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWSortedLightInfo)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, VisibleLightHashHistory)
		SHADER_PARAMETER(FIntPoint, HistoryVisibleLightHashViewMinInTiles)
		SHADER_PARAMETER(FIntPoint, HistoryVisibleLightHashViewSizeInTiles)
		SHADER_PARAMETER(float, TransmissionSampleWeight)
		SHADER_PARAMETER(FIntVector, VolumeVisibleLightHashTileSize)
		SHADER_PARAMETER(FIntVector, HistoryVolumeVisibleLightHashViewSizeInTiles)
	END_SHADER_PARAMETER_STRUCT()

	static int32 GetGroupSize()
	{
		return 512;
	}

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		if (PermutationVector.Get<FIsVisualizingVolume>())
		{
			PermutationVector.Set<FInputType>(int32(EMegaLightsInput::GBuffer));
		}
		else
		{
			PermutationVector.Set<FTranslucencyLightingVolume>(false);
		}

		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		const FPermutationDomain PermutationVector(Parameters.PermutationId);

		if (RemapPermutation(PermutationVector) != PermutationVector)
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return EShaderPermutationPrecacheRequest::NotPrecached;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
		OutEnvironment.CompilerFlags.Add(CFLAG_WaveOperations);
	}
};

IMPLEMENT_GLOBAL_SHADER(FMegaLightsSortLightsForDisplayCS, "/Engine/Private/MegaLights/MegaLightsLightComplexity.usf", "MegaLightsSortLightsForDisplayCS", SF_Compute);

class FMegaLightsIntersectDisplayedLightsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMegaLightsIntersectDisplayedLightsCS)
	SHADER_USE_PARAMETER_STRUCT(FMegaLightsIntersectDisplayedLightsCS, FGlobalShader)

	using FPermutationDomain = TShaderPermutationDomain<>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, FrozenLightData)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, SortedLightInfo)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWSelectedLightIndex)
		SHADER_PARAMETER(uint32, MaxNumLightsToDisplay)
		SHADER_PARAMETER(int32, bUseFrozenLightData)
	END_SHADER_PARAMETER_STRUCT()

	static int32 GetGroupSize()
	{
		return 64;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return EShaderPermutationPrecacheRequest::NotPrecached;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
		OutEnvironment.CompilerFlags.Add(CFLAG_WaveOperations);
	}
};

IMPLEMENT_GLOBAL_SHADER(FMegaLightsIntersectDisplayedLightsCS, "/Engine/Private/MegaLights/MegaLightsLightComplexity.usf", "MegaLightsIntersectDisplayedLightsCS", SF_Compute);

class FMegaLightsDisplayLightsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMegaLightsDisplayLightsCS)
	SHADER_USE_PARAMETER_STRUCT(FMegaLightsDisplayLightsCS, FGlobalShader)

	class FInputType : SHADER_PERMUTATION_INT("INPUT_TYPE", int32(EMegaLightsInput::Count));
	class FTranslucencyLightingVolume : SHADER_PERMUTATION_BOOL("TRANSLUCENCY_LIGHTING_VOLUME");
	using FPermutationDomain = TShaderPermutationDomain<FInputType, FTranslucencyLightingVolume>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsVolumeParameters, MegaLightsVolumeParameters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWGeneralPurposeState)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWGeneralPurposeFeedback)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWFrozenLightData)
		SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, FrozenLightData)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, SortedLightInfo)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, SelectedLightIndexBuffer)
		SHADER_PARAMETER(uint32, MaxNumLightsToDisplay)
		SHADER_PARAMETER(int32, bOutputFrozenLightData)
		SHADER_PARAMETER(int32, bUseFrozenLightData)
		SHADER_PARAMETER(int32, bIsVisualizingVolume)
	END_SHADER_PARAMETER_STRUCT()

	static int32 GetGroupSize()
	{
		return 64;
	}

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		if (PermutationVector.Get<FTranslucencyLightingVolume>())
		{
			PermutationVector.Set<FInputType>(int32(EMegaLightsInput::GBuffer));
		}

		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		const FPermutationDomain PermutationVector(Parameters.PermutationId);

		if (RemapPermutation(PermutationVector) != PermutationVector)
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return EShaderPermutationPrecacheRequest::NotPrecached;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
		OutEnvironment.CompilerFlags.Add(CFLAG_WaveOperations);
	}
};

IMPLEMENT_GLOBAL_SHADER(FMegaLightsDisplayLightsCS, "/Engine/Private/MegaLights/MegaLightsLightComplexity.usf", "MegaLightsDisplayLightsCS", SF_Compute);

class FMegaLightsEstimateLightSamplingCostCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMegaLightsEstimateLightSamplingCostCS)
	SHADER_USE_PARAMETER_STRUCT(FMegaLightsEstimateLightSamplingCostCS, FGlobalShader)

	class FInputType : SHADER_PERMUTATION_INT("INPUT_TYPE", int32(EMegaLightsInput::Count));
	using FPermutationDomain = TShaderPermutationDomain<FInputType>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWLightSamplingCostEstimate)
		SHADER_PARAMETER(int32, bUseHairComplexTransmittance)
	END_SHADER_PARAMETER_STRUCT()

	static int32 GetGroupSize()
	{
		return 8;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return EShaderPermutationPrecacheRequest::NotPrecached;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FMegaLightsEstimateLightSamplingCostCS, "/Engine/Private/MegaLights/MegaLightsLightComplexity.usf", "MegaLightsEstimateLightSamplingCostCS", SF_Compute);

class FMegaLightsPostProcessVisualizePS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMegaLightsPostProcessVisualizePS)
	SHADER_USE_PARAMETER_STRUCT(FMegaLightsPostProcessVisualizePS, FGlobalShader)

	using FPermutationDomain = TShaderPermutationDomain<>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
		SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrintUniformBuffer)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InputSceneColor)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, LightSamplingCostEstimateTexture)
		SHADER_PARAMETER(FScreenTransform, OutputToInputPixelPos)
		SHADER_PARAMETER(float, MaxLightSamplingCost)
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return EShaderPermutationPrecacheRequest::NotPrecached;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
	}
};

IMPLEMENT_GLOBAL_SHADER(FMegaLightsPostProcessVisualizePS, "/Engine/Private/MegaLights/MegaLightsLightComplexity.usf", "MegaLightsPostProcessVisualizePS", SF_Pixel);

static void ProcessDumpLightDataReadback(TConstArrayView<uint8> FrozenLightData, uint32 LightNameDwords)
{
	// Keep in-sync with MegaLightsLightComplexity.usf
	const uint32 GlobalFrozenDataBytes = MegaLightsLightComplexity::GlobalFrozenDataDwords * sizeof(uint32);
	const uint32 LightFrozenDataNoNameDwords = MegaLightsLightComplexity::LightFrozenDataExceptNameDwords;
	const uint32 LightFrozenDataBytes = (LightFrozenDataNoNameDwords + LightNameDwords) * sizeof(uint32);
	const uint8* ReadPtr = FrozenLightData.GetData();

	const uint32 NumLights = *(const uint32*)ReadPtr;
	ReadPtr += sizeof(uint32);

	check(GlobalFrozenDataBytes + NumLights * LightFrozenDataBytes <= (uint32)FrozenLightData.Num());

	const float LightTargetPDFSum = *(const float*)ReadPtr;
	ReadPtr += sizeof(float);
	// Skip pixel translated world position (3 dwords)
	ReadPtr += sizeof(FVector3f);
	const FVector3f PreViewTranslationLow = *(const FVector3f*)ReadPtr;
	ReadPtr += sizeof(FVector3f);
	const FVector3f PreViewTranslationHigh = *(const FVector3f*)ReadPtr;
	ReadPtr += sizeof(FVector3f);
	// Skip voxel vertices (24 dwords)
	ReadPtr += sizeof(FVector3f) * 8;
	// Skip volume vertices (24 dwords)
	ReadPtr += sizeof(FVector3f) * 8;

	check(ReadPtr - FrozenLightData.GetData() == GlobalFrozenDataBytes);

	const FVector3d PreViewTranslation = FDFVector3(PreViewTranslationHigh, PreViewTranslationLow).GetVector3d();

	uint32 MaxLightNameLen = 0;
	for (uint32 LightIndex = 0; LightIndex < NumLights; ++LightIndex)
	{
		const uint8* LightDataBase = ReadPtr + LightIndex * LightFrozenDataBytes;
		const uint32 Packed0 = *(const uint32*)LightDataBase;
		const bool bValid = (Packed0 & 0x80000000) == 0;

		if (bValid)
		{
			const ANSICHAR* LightName = (const ANSICHAR*)(LightDataBase + LightFrozenDataNoNameDwords * sizeof(uint32));
			const uint32 LightNameLen = FCStringAnsi::Strnlen(LightName, LightNameDwords * sizeof(uint32));
			MaxLightNameLen = FMath::Max(LightNameLen, MaxLightNameLen);
		}
	}

	FString DumpOutputStr;
	DumpOutputStr.Appendf(TEXT("Dumping MegaLights VisualizeLightComplexity light list...\nNumLights: %d\n"), NumLights);
	DumpOutputStr.Appendf(TEXT("%*s | WeightRatio (Weight) | Visibility | LightType | WorldPosition or Direction\n"), MaxLightNameLen, TEXT("LightName"));

	for (uint32 LightIndex = 0; LightIndex < NumLights; ++LightIndex)
	{
		const uint8* LightDataStart = ReadPtr;

		const uint32 Packed0 = *(const uint32*)ReadPtr;
		ReadPtr += sizeof(uint32);
		const float LightTargetPDF = *(const float*)ReadPtr;
		ReadPtr += sizeof(float);
		const FVector3f LightTWPositionOrDirection = *(const FVector3f*)ReadPtr;
		ReadPtr += sizeof(FVector3f);
		const FVector3f LightColor = *(const FVector3f*)ReadPtr;
		ReadPtr += sizeof(FVector3f);
		const ANSICHAR* LightName = (const ANSICHAR*)ReadPtr;
		ReadPtr += LightNameDwords * sizeof(uint32);
		check(ReadPtr - LightDataStart == LightFrozenDataBytes);

		const bool bValid = (Packed0 & 0x80000000) == 0;

		if (bValid)
		{
			const uint32 LightType = (Packed0 >> 8) & 0x3;
			const float LightVisibility = (Packed0 & 0xFF) / 255.0f;
			const FVector3d LightWorldPositionOrDirection = LightType != LIGHT_TYPE_DIRECTIONAL ? FVector3d(LightTWPositionOrDirection) - PreViewTranslation : FVector3d(LightTWPositionOrDirection);

			const FString WeightAndRatioStr = FString::Printf(TEXT("%.3f (%.3f)"), LightTargetPDF / FMath::Max(LightTargetPDFSum, 1e-6f), LightTargetPDF);

			const TCHAR* LightTypeStr;
			switch (LightType)
			{
			case LIGHT_TYPE_DIRECTIONAL:
				LightTypeStr = TEXT("Dir");
				break;
			case LIGHT_TYPE_POINT:
				LightTypeStr = TEXT("Point");
				break;
			case LIGHT_TYPE_SPOT:
				LightTypeStr = TEXT("Spot");
				break;
			case LIGHT_TYPE_RECT:
				LightTypeStr = TEXT("Rect");
				break;
			default:
				LightTypeStr = TEXT("Unknown");
				break;
			}

			DumpOutputStr.Appendf(TEXT("%*s ,%*s , %10.3f , %9s , (%.3f, %.3f, %.3f)\n"),
				MaxLightNameLen,
				StringCast<TCHAR>(LightName).Get(),
				21,
				*WeightAndRatioStr,
				LightVisibility,
				LightTypeStr,
				LightWorldPositionOrDirection.X,
				LightWorldPositionOrDirection.Y,
				LightWorldPositionOrDirection.Z);
		}
	}

	UE_LOGF(LogRenderer, Log, "%ls", *DumpOutputStr);
}

void FMegaLightsViewContext::DispatchDebugTileClassificationPasses(ERDGPassFlags ComputePassFlags)
{
	if (DebugTileClassificationMode != 0 && ((DebugTileClassificationMode - 1) / 2) == (uint32)InputType)
	{
		FMegaLightsDebugCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsDebugCS::FParameters>();
		PassParameters->IndirectArgs = DownsampledTileIndirectArgs;
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->TileAllocator = GraphBuilder.CreateSRV(TileAllocator);
		PassParameters->TileData = GraphBuilder.CreateSRV(TileData);
		PassParameters->DownsampledTileAllocator = GraphBuilder.CreateSRV(DownsampledTileAllocator);
		PassParameters->DownsampledTileData = GraphBuilder.CreateSRV(DownsampledTileData);
		PassParameters->DebugTileClassificationMode = DebugTileClassificationMode;

		auto ComputeShader = View.ShaderMap->GetShader<FMegaLightsDebugCS>();
		const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(ViewSizeInTiles.X * ViewSizeInTiles.Y, FMegaLightsDebugCS::GetGroupSize());

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("Debug"),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			GroupCount);
	}
}

void FMegaLightsViewContext::DispatchVisualizeLightsPasses(ERDGPassFlags ComputePassFlags)
{
	if (VisualizeLightComplexityTarget != EMegaLightsDebugTarget::None && View.ForwardLightingResources.ForwardLightUniformParameters)
	{
		const FForwardLightUniformParameters& ForwardLightUniforms = *View.ForwardLightingResources.ForwardLightUniformParameters;

		const uint32 GlobalFrozenDataDwords = MegaLightsLightComplexity::GlobalFrozenDataDwords;
		const uint32 LightFrozenDataDwords = MegaLightsLightComplexity::LightFrozenDataExceptNameDwords + ForwardLightUniforms.LightNameDwords;
		const uint32 MaxNumLightsToSort = ForwardLightUniforms.MaxCulledLightsPerCell + ForwardLightUniforms.NumDirectionalLights;
		const bool bAnyLightInView = MaxNumLightsToSort > 0;
		const bool bIsVisualizingVolume = (VisualizeLightComplexityTarget & EMegaLightsDebugTarget::VolumeMask) != EMegaLightsDebugTarget::None;
		const bool bTranslucencyLightingVolume = VisualizeLightComplexityTarget == EMegaLightsDebugTarget::TranslucencyLightingVolume;
		const int32 PrevTLVCascadeIndex = MegaLightsTranslucencyVolumeParameters.TranslucencyVolumeCascadeIndex;

		FRDGBufferRef GeneralPurposeStateBuffer = nullptr;
		FRDGBufferRef FrozenLightDataBufferIn = nullptr;
		FRDGBufferRef FrozenLightDataBufferOut = nullptr;
		bool bOutputFrozenLightData = false;
		bool bUseFrozenLightData = false;
		FMegaLightsViewState::FResources* MegaLightsViewState = nullptr;

		if (View.ViewState)
		{
			MegaLightsViewState = &MegaLights::GetViewState(View, InputType);

			if (bVisualizeLightComplexityDump && MegaLightsViewState->DumpLightDataReadback)
			{
				// Skip the new dump request because there is a pending request
				bVisualizeLightComplexityDump = false;
			}

			if (bVisualizeLightComplexityFrozen || bVisualizeLightComplexityDump)
			{
				if (MegaLightsViewState->FrozenLightData)
				{
					FrozenLightDataBufferIn = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState->FrozenLightData);
					bUseFrozenLightData = true;
				}
				else if (bAnyLightInView)
				{
					const uint32 BufferSizeInBytes = (GlobalFrozenDataDwords + LightFrozenDataDwords * MaxNumLightsToSort) * sizeof(uint32);
					FrozenLightDataBufferOut = GraphBuilder.CreateBuffer(
						FRDGBufferDesc::CreateByteAddressDesc(BufferSizeInBytes),
						MEGALIGHTS_RESOURCE_NAME("LightComplexity.FrozenLightData"));
					bOutputFrozenLightData = true;
				}
			}
			else
			{
				MegaLightsViewState->FrozenLightData = nullptr;
			}

			if (MegaLightsViewState->GeneralPurposeState)
			{
				GeneralPurposeStateBuffer = GraphBuilder.RegisterExternalBuffer(MegaLightsViewState->GeneralPurposeState);
			}

			if (MegaLightsViewState->GeneralPurposeReadback)
			{
				const uint32* Data = (const uint32*)MegaLightsViewState->GeneralPurposeReadback->LockReadback();
				if (Data)
				{
					if ((Data[0] & 0x1) != 0)
					{
						const bool bInnerTLVSelected = (Data[0] & 0x2) != 0;
						MegaLightsTranslucencyVolumeParameters.TranslucencyVolumeCascadeIndex = bInnerTLVSelected ? 0 : 1;
					}
				}
				MegaLightsViewState->GeneralPurposeReadback->UnlockReadback();
			}
		}

		FRDGBufferRef GeneralPurposeFeedbackBuffer = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateByteAddressDesc(MegaLightsLightComplexity::GeneralPurposeFeedbackDwords * sizeof(uint32)),
			MEGALIGHTS_RESOURCE_NAME("LightComplexity.GeneralPurposeFeedback"));

		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(GeneralPurposeFeedbackBuffer), 0);

		FRDGBufferRef SortedLightInfoBuffer = nullptr;
		if (!bUseFrozenLightData && bAnyLightInView)
		{
			// Keep in-sync with MegaLightsLightComplexity.usf
			const uint32 BufferSizeInDwords = 2 + MaxNumLightsToSort * 2;

			SortedLightInfoBuffer = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), BufferSizeInDwords),
				MEGALIGHTS_RESOURCE_NAME("LightComplexity.SortedLightInfo"));
		}

		if (!GeneralPurposeStateBuffer)
		{
			GeneralPurposeStateBuffer = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateByteAddressDesc(MegaLightsLightComplexity::GeneralPurposeStateDwords * sizeof(uint32)),
				MEGALIGHTS_RESOURCE_NAME("LightComplexity.GeneralPurposeState"));

			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(GeneralPurposeStateBuffer), 0);
		}

		if (!FrozenLightDataBufferIn)
		{
			FrozenLightDataBufferIn = CreateByteAddressBuffer(GraphBuilder, MEGALIGHTS_RESOURCE_NAME("LightComplexity.DummyFrozenLightDataIn"), TArray<uint32>({ 0 }));
		}
		if (!FrozenLightDataBufferOut)
		{
			FrozenLightDataBufferOut = CreateByteAddressBuffer(GraphBuilder, MEGALIGHTS_RESOURCE_NAME("LightComplexity.DummyFrozenLightDataOut"), TArray<uint32>({ 0 }));
		}
		if (!SortedLightInfoBuffer)
		{
			SortedLightInfoBuffer = CreateStructuredBuffer(GraphBuilder, MEGALIGHTS_RESOURCE_NAME("LightComplexity.DummySortedLightInfo"), TArray<uint32>({ 0 }));
		}

		if (!bUseFrozenLightData && bAnyLightInView)
		{
			FRDGBufferRef LocalVisibleLightHashHistory = VisibleLightHashHistory;
			if (bTranslucencyLightingVolume)
			{
				LocalVisibleLightHashHistory = TranslucencyVolumeVisibleLightHashHistory[MegaLightsTranslucencyVolumeParameters.TranslucencyVolumeCascadeIndex];
			}
			else if (bIsVisualizingVolume)
			{
				LocalVisibleLightHashHistory = VolumeVisibleLightHashHistory;
			}

			const bool bHasLightVisibilityHistory = LocalVisibleLightHashHistory != nullptr;

			FMegaLightsSortLightsForDisplayCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsSortLightsForDisplayCS::FParameters>();
			PassParameters->MegaLightsParameters = MegaLightsParameters;
			PassParameters->HistoryScreenParameters = HistoryScreenParameters;
			PassParameters->RWGeneralPurposeState = GraphBuilder.CreateUAV(GeneralPurposeStateBuffer);
			PassParameters->RWSortedLightInfo = GraphBuilder.CreateUAV(SortedLightInfoBuffer, PF_R32_UINT);
			PassParameters->VisibleLightHashHistory = bHasLightVisibilityHistory ? GraphBuilder.CreateSRV(LocalVisibleLightHashHistory) : nullptr;
			PassParameters->HistoryVisibleLightHashViewMinInTiles = HistoryVisibleLightHashViewMinInTiles;
			PassParameters->HistoryVisibleLightHashViewSizeInTiles = HistoryVisibleLightHashViewSizeInTiles;
			PassParameters->TransmissionSampleWeight = MegaLights::GetTransmissionSampleWeight();

			if (bTranslucencyLightingVolume)
			{
				PassParameters->MegaLightsVolumeParameters = MegaLightsTranslucencyVolumeParameters;
				PassParameters->VolumeVisibleLightHashTileSize = TranslucencyVolumeVisibleLightHashTileSize;
				PassParameters->HistoryVolumeVisibleLightHashViewSizeInTiles = HistoryTranslucencyVolumeVisibleLightHashSizeInTiles;
			}
			else
			{
				PassParameters->MegaLightsVolumeParameters = MegaLightsVolumeParameters;
				PassParameters->VolumeVisibleLightHashTileSize = VolumeVisibleLightHashTileSize;
				PassParameters->HistoryVolumeVisibleLightHashViewSizeInTiles = HistoryVolumeVisibleLightHashViewSizeInTiles;
			}

			FMegaLightsSortLightsForDisplayCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FMegaLightsSortLightsForDisplayCS::FInputType>(int32(InputType));
			PermutationVector.Set<FMegaLightsSortLightsForDisplayCS::FIsVisualizingVolume>(bIsVisualizingVolume);
			PermutationVector.Set<FMegaLightsSortLightsForDisplayCS::FTranslucencyLightingVolume>(bTranslucencyLightingVolume);
			PermutationVector.Set<FMegaLightsSortLightsForDisplayCS::FHasLightVisibilityHistory>(bHasLightVisibilityHistory);
			auto ComputeShader = View.ShaderMap->GetShader<FMegaLightsSortLightsForDisplayCS>(PermutationVector);

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("SortLightsForDisplay"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				FIntVector(1, 1, 1));
		}

		const uint32 MaxNumLightsToDisplay = FMath::Max(CVarMegaLightsVisualizeLightComplexityMaxLightsToDisplay.GetValueOnRenderThread(), 0);
		FRDGBufferRef SelectedLightIndexBuffer = CreateStructuredBuffer(GraphBuilder, MEGALIGHTS_RESOURCE_NAME("LightComplexity.SelectedLightIndexBuffer"), TArray<uint32>({ 0 }));

		if (bUseFrozenLightData || bAnyLightInView)
		{
			FMegaLightsIntersectDisplayedLightsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsIntersectDisplayedLightsCS::FParameters>();
			PassParameters->MegaLightsParameters = MegaLightsParameters;
			PassParameters->FrozenLightData = GraphBuilder.CreateSRV(FrozenLightDataBufferIn);
			PassParameters->SortedLightInfo = GraphBuilder.CreateSRV(SortedLightInfoBuffer);
			PassParameters->RWSelectedLightIndex = GraphBuilder.CreateUAV(SelectedLightIndexBuffer);
			PassParameters->MaxNumLightsToDisplay = MaxNumLightsToDisplay;
			PassParameters->bUseFrozenLightData = bUseFrozenLightData;

			FMegaLightsIntersectDisplayedLightsCS::FPermutationDomain PermutationVector;
			auto ComputeShader = View.ShaderMap->GetShader<FMegaLightsIntersectDisplayedLightsCS>(PermutationVector);

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("IntersectDisplayedLights"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				FIntVector(1, 1, 1));
		}

		if (bUseFrozenLightData || bAnyLightInView)
		{
			FMegaLightsDisplayLightsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsDisplayLightsCS::FParameters>();
			PassParameters->MegaLightsParameters = MegaLightsParameters;
			PassParameters->MegaLightsVolumeParameters = bTranslucencyLightingVolume ? MegaLightsTranslucencyVolumeParameters : MegaLightsVolumeParameters;
			PassParameters->RWGeneralPurposeState = GraphBuilder.CreateUAV(GeneralPurposeStateBuffer);
			PassParameters->RWGeneralPurposeFeedback = GraphBuilder.CreateUAV(GeneralPurposeFeedbackBuffer);
			PassParameters->RWFrozenLightData = GraphBuilder.CreateUAV(FrozenLightDataBufferOut);
			PassParameters->FrozenLightData = GraphBuilder.CreateSRV(FrozenLightDataBufferIn);
			PassParameters->SortedLightInfo = GraphBuilder.CreateSRV(SortedLightInfoBuffer, PF_R32_UINT);
			PassParameters->SelectedLightIndexBuffer = GraphBuilder.CreateSRV(SelectedLightIndexBuffer);
			PassParameters->MaxNumLightsToDisplay = MaxNumLightsToDisplay;
			PassParameters->bOutputFrozenLightData = bOutputFrozenLightData;
			PassParameters->bUseFrozenLightData = bUseFrozenLightData;
			PassParameters->bIsVisualizingVolume = bIsVisualizingVolume;

			FMegaLightsDisplayLightsCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FMegaLightsDisplayLightsCS::FInputType>(int32(InputType));
			PermutationVector.Set<FMegaLightsDisplayLightsCS::FTranslucencyLightingVolume>(bTranslucencyLightingVolume);
			auto ComputeShader = View.ShaderMap->GetShader<FMegaLightsDisplayLightsCS>(PermutationVector);

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("DisplayLights"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				FIntVector(1, 1, 1));
		}

		if (LightSamplingCostEstimate != nullptr)
		{
			FMegaLightsEstimateLightSamplingCostCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsEstimateLightSamplingCostCS::FParameters>();
			PassParameters->MegaLightsParameters = MegaLightsParameters;
			PassParameters->RWLightSamplingCostEstimate = GraphBuilder.CreateUAV(LightSamplingCostEstimate);
			PassParameters->bUseHairComplexTransmittance = bUseHairComplexTransmittance;

			FMegaLightsEstimateLightSamplingCostCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FMegaLightsEstimateLightSamplingCostCS::FInputType>(int32(InputType));
			auto ComputeShader = View.ShaderMap->GetShader<FMegaLightsEstimateLightSamplingCostCS>(PermutationVector);

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("EstimateLightSamplingCost"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				FComputeShaderUtils::GetGroupCount(View.ViewRect.Size(), FMegaLightsEstimateLightSamplingCostCS::GetGroupSize()));
		}

		if (bVisualizeLightComplexityDump && (bUseFrozenLightData || bOutputFrozenLightData))
		{
			FRDGBufferRef LightDataBuffer = bUseFrozenLightData ? FrozenLightDataBufferIn : FrozenLightDataBufferOut;

			check(MegaLightsViewState && !MegaLightsViewState->DumpLightDataReadback);
			FRHIGPUBufferReadback* LightDataReadback = new FRHIGPUBufferReadback("MegaLights.LightDataReadback");
			MegaLightsViewState->DumpLightDataReadback = LightDataReadback;
			MegaLightsViewState->PendingDumpLightDataReadbackSize = LightDataBuffer->GetSize();

			AddReadbackBufferPass(GraphBuilder, RDG_EVENT_NAME("LightDataReadback"), LightDataBuffer,
				[LightDataReadback, LightDataBuffer](FRDGAsyncTask, FRHICommandList& RHICmdList)
				{
					LightDataReadback->EnqueueCopy(RHICmdList, LightDataBuffer->GetRHI());
				});
		}

		if (bOutputFrozenLightData && bVisualizeLightComplexityFrozen)
		{
			check(MegaLightsViewState);
			GraphBuilder.QueueBufferExtraction(FrozenLightDataBufferOut, &MegaLightsViewState->FrozenLightData);
		}

		if (MegaLightsViewState)
		{
			GraphBuilder.QueueBufferExtraction(GeneralPurposeStateBuffer, &MegaLightsViewState->GeneralPurposeState);

			if (!MegaLightsViewState->GeneralPurposeReadback)
			{
				MegaLightsViewState->GeneralPurposeReadback = new FGPUBufferReadbackCollection;
			}

			MegaLightsViewState->GeneralPurposeReadback->EnqueueReadback(GraphBuilder, GeneralPurposeFeedbackBuffer, GeneralPurposeFeedbackBuffer->GetSize());

			if (MegaLightsViewState->DumpLightDataReadback && MegaLightsViewState->DumpLightDataReadback->IsReady())
			{
				const uint8* LockedPtr = (const uint8*)MegaLightsViewState->DumpLightDataReadback->Lock(MegaLightsViewState->PendingDumpLightDataReadbackSize);
				const TConstArrayView<uint8> DumpLightDataView(LockedPtr, MegaLightsViewState->PendingDumpLightDataReadbackSize);

				ProcessDumpLightDataReadback(DumpLightDataView, ForwardLightUniforms.LightNameDwords);

				MegaLightsViewState->DumpLightDataReadback->Unlock();
				delete MegaLightsViewState->DumpLightDataReadback;
				MegaLightsViewState->DumpLightDataReadback = nullptr;
				MegaLightsViewState->PendingDumpLightDataReadbackSize = 0;
			}

			MegaLightsViewState->LastVisualizeLightComplexityTarget = VisualizeLightComplexityTarget;
		}

		MegaLightsTranslucencyVolumeParameters.TranslucencyVolumeCascadeIndex = PrevTLVCascadeIndex;
	}
	else
	{
		if (View.ViewState)
		{
			FMegaLightsViewState::FResources& MegaLightsViewState = MegaLights::GetViewState(View, InputType);

			MegaLightsViewState.GeneralPurposeState = nullptr;
			MegaLightsViewState.FrozenLightData = nullptr;
			MegaLightsViewState.LastVisualizeLightComplexityTarget = EMegaLightsDebugTarget::None;

			if (MegaLightsViewState.GeneralPurposeReadback)
			{
				delete MegaLightsViewState.GeneralPurposeReadback;
				MegaLightsViewState.GeneralPurposeReadback = nullptr;
			}

			if (MegaLightsViewState.DumpLightDataReadback && MegaLightsViewState.DumpLightDataReadback->IsReady())
			{
				delete MegaLightsViewState.DumpLightDataReadback;
				MegaLightsViewState.DumpLightDataReadback = nullptr;
				MegaLightsViewState.PendingDumpLightDataReadbackSize = 0;
			}
		}
	}
}

void FMegaLightsViewContext::VisualizeLightComplexity(const FScreenPassRenderTarget& OutputSceneColor, const FScreenPassTexture& InputSceneColor)
{
	FScreenPassTexture ScreenPassCostEstimate(LightSamplingCostEstimate, View.ViewRect);

	FMegaLightsPostProcessVisualizePS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsPostProcessVisualizePS::FParameters>();
	PassParameters->ViewUniformBuffer = View.ViewUniformBuffer;
	ShaderPrint::SetParameters(GraphBuilder, View.ShaderPrintData, PassParameters->ShaderPrintUniformBuffer);
	PassParameters->InputSceneColor = InputSceneColor.Texture;
	PassParameters->LightSamplingCostEstimateTexture = ScreenPassCostEstimate.Texture;
	PassParameters->OutputToInputPixelPos =
		FScreenTransform::ChangeTextureBasisFromTo(InputSceneColor, FScreenTransform::ETextureBasis::TexelPosition, FScreenTransform::ETextureBasis::ViewportUV) *
		FScreenTransform::ChangeTextureBasisFromTo(ScreenPassCostEstimate, FScreenTransform::ETextureBasis::ViewportUV, FScreenTransform::ETextureBasis::TexelPosition);
	PassParameters->MaxLightSamplingCost = CVarMegaLightsVisualizeLightComplexityMaxLightSamplingCost.GetValueOnRenderThread();
	PassParameters->RenderTargets[0] = FRenderTargetBinding(OutputSceneColor.Texture, ERenderTargetLoadAction::ELoad);

	FMegaLightsPostProcessVisualizePS::FPermutationDomain PermutationVector;
	auto PixelShader = View.ShaderMap->GetShader<FMegaLightsPostProcessVisualizePS>(PermutationVector);

	FRHIBlendState* BlendState = TStaticBlendState<CW_RGB, BO_Add, BF_One, BF_SourceAlpha, BO_Add, BF_Zero, BF_One>::GetRHI();

	FPixelShaderUtils::AddFullscreenPass<FMegaLightsPostProcessVisualizePS>(
		GraphBuilder,
		View.ShaderMap,
		RDG_EVENT_NAME("MegaLightsVisualizeLightSamplingCost"),
		PixelShader,
		PassParameters,
		OutputSceneColor.ViewRect,
		BlendState);
}

FScreenPassTexture MegaLights::AddVisualizeLightComplexityPostProcessingPass(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	int32 ViewIndex,
	const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries,
	FScreenPassTexture InputSceneColor,
	FScreenPassRenderTarget OverrideOutput)
{
	FScreenPassRenderTarget OutputSceneColor = OverrideOutput;
	if (!OutputSceneColor.IsValid())
	{
		OutputSceneColor = FScreenPassRenderTarget::CreateFromInput(GraphBuilder, InputSceneColor, View.GetOverwriteLoadAction(), TEXT("SceneColorWithMegaLightsSamplingCost"));
	}

	MegaLightsFrameTemporaries->ViewContexts[ViewIndex].VisualizeLightComplexity(OutputSceneColor, InputSceneColor);

	return OutputSceneColor;
}

FGPUBufferReadbackCollection::FGPUBufferReadbackCollection()
	: WritePtr(0)
	, NumPending(0)
	, LockedReadback(nullptr)
{
	for (uint32 Index = 0; Index < NumReadbacks; ++Index)
	{
		ReadbackSizeArray[Index] = 0;
		Readbacks[Index] = nullptr;
	}
}

FGPUBufferReadbackCollection::~FGPUBufferReadbackCollection()
{
	for (uint32 Index = 0; Index < NumReadbacks; ++Index)
	{
		if (Readbacks[Index])
		{
			delete Readbacks[Index];
			Readbacks[Index] = nullptr;
		}
	}
}

bool FGPUBufferReadbackCollection::EnqueueReadback(FRDGBuilder& GraphBuilder, FRDGBufferRef BufferToReadback, uint32 SizeInBytes)
{
	check(NumPending <= NumReadbacks && !LockedReadback);

	if (NumPending == NumReadbacks)
	{
		LockReadback();
		UnlockReadback();

		if (NumPending == NumReadbacks)
		{
			return false;
		}
	}

	if (!Readbacks[WritePtr])
	{
		Readbacks[WritePtr] = new FRHIGPUBufferReadback("CollectionReadback");
	}

	FRHIGPUBufferReadback* Readback = Readbacks[WritePtr];
	AddReadbackBufferPass(GraphBuilder, RDG_EVENT_NAME("CollectionReadback"), BufferToReadback,
		[Readback, BufferToReadback](FRDGAsyncTask, FRHICommandList& RHICmdList)
		{
			Readback->EnqueueCopy(RHICmdList, BufferToReadback->GetRHI());
		});

	ReadbackSizeArray[WritePtr] = SizeInBytes;
	WritePtr = (WritePtr + 1) % NumReadbacks;
	++NumPending;

	return true;
}

void* FGPUBufferReadbackCollection::LockReadback()
{
	check(!LockedReadback);
	uint32 SizeInBytes = 0;

	while (NumPending > 0)
	{
		const int32 ReadPtr = (WritePtr + NumReadbacks - NumPending) % NumReadbacks;
		FRHIGPUBufferReadback* Readback = Readbacks[ReadPtr];
		check(Readback);

		if (!Readback->IsReady())
		{
			break;
		}

		LockedReadback = Readback;
		SizeInBytes = ReadbackSizeArray[ReadPtr];
		--NumPending;
	}

	return LockedReadback ? LockedReadback->Lock(SizeInBytes) : nullptr;
}

void FGPUBufferReadbackCollection::UnlockReadback()
{
	if (LockedReadback)
	{
		LockedReadback->Unlock();
		LockedReadback = nullptr;
	}
}
