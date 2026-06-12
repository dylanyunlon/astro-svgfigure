// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BlueNoise.h"
#include "MegaLightsDefinitions.h"
#include "VolumetricCloudRendering.h"
#include "StochasticLighting/StochasticLighting.h"
#include "Lumen/LumenHardwareRayTracingCommon.h"
#include "MegaLightsViewState.h"

#if UE_BUILD_SHIPPING
#define MEGALIGHTS_RESOURCE_NAME(BaseName) TEXT("MegaLights." BaseName)
#else
#define MEGALIGHTS_RESOURCE_NAME(BaseName) \
    ([](EMegaLightsInput InType) -> const TCHAR* { \
        switch (InType) { \
        case EMegaLightsInput::GBuffer:                return TEXT("MegaLights." BaseName); \
        case EMegaLightsInput::HairStrands:            return TEXT("MegaLights.HairStrands." BaseName); \
        case EMegaLightsInput::FrontLayerTranslucency: return TEXT("MegaLights.FrontLayerTranslucency." BaseName); \
        default: checkNoEntry();                       return TEXT("MegaLights." BaseName); \
        } \
    })(InputType)
#endif

BEGIN_SHADER_PARAMETER_STRUCT(FMegaLightsParameters, )
	SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FHairStrandsViewUniformParameters, HairStrands)
	SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrintUniformBuffer)
	SHADER_PARAMETER_STRUCT_INCLUDE(FSceneTextureParameters, SceneTextures)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTexturesStruct)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSubstrateGlobalUniformParameters, Substrate)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FForwardLightUniformParameters, ForwardLightStruct)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(LightFunctionAtlas::FLightFunctionAtlasGlobalParameters, LightFunctionAtlas)
	SHADER_PARAMETER_STRUCT_INCLUDE(FLightCloudTransmittanceParameters, CloudShadow)
	SHADER_PARAMETER_STRUCT_INCLUDE(FSceneLightingChannelParameters, LightingChannelParameters)
	SHADER_PARAMETER_STRUCT_INCLUDE(FFrontLayerTranslucencyGBufferParameters, FrontLayerTranslucencyGBufferParameters)
	SHADER_PARAMETER_STRUCT_REF(FBlueNoise, BlueNoise)
	SHADER_PARAMETER_TEXTURE(Texture2D, PreIntegratedGF)
	SHADER_PARAMETER_SAMPLER(SamplerState, PreIntegratedGFSampler)
	SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint>, MegaLightsTileBitmask)
	SHADER_PARAMETER(FIntPoint, SampleViewMin)
	SHADER_PARAMETER(FIntPoint, SampleViewSize)
	SHADER_PARAMETER(FIntPoint, DownsampledViewMin)
	SHADER_PARAMETER(FIntPoint, DownsampledViewSize)
	SHADER_PARAMETER(FIntPoint, NumSamplesPerPixel)
	SHADER_PARAMETER(FIntPoint, NumSamplesPerPixelDivideShift)
	SHADER_PARAMETER(FVector2f, DownsampledBufferInvSize)	
	SHADER_PARAMETER(FIntPoint, DownsampleFactor)
	SHADER_PARAMETER(uint32, MegaLightsStateFrameIndex)
	SHADER_PARAMETER(uint32, StochasticLightingStateFrameIndex)
	SHADER_PARAMETER(float, MinSampleWeight)
	SHADER_PARAMETER(float, MinSampleWeightEstimate)
	SHADER_PARAMETER(int32, TileDataStride)
	SHADER_PARAMETER(int32, DownsampledTileDataStride)
	SHADER_PARAMETER(FIntPoint, DebugCursorPosition)
	SHADER_PARAMETER(int32, DebugLightId)
	SHADER_PARAMETER(int32, DebugVisualizeLight)
	SHADER_PARAMETER(int32, DebugVisualizeTraces)
	SHADER_PARAMETER(int32, UseIESProfiles)
	SHADER_PARAMETER(int32, UseLightFunctionAtlas)
	SHADER_PARAMETER(int32, bCloudShadowEnabled)
	SHADER_PARAMETER(int32, bDirectionalLightPerPixelTransmittanceEnabled)
	SHADER_PARAMETER(int32, SelectedForwardDirectionalLightIndex)
	SHADER_PARAMETER(int32, FrontLayerTranslucencySpecularOnly)
	SHADER_PARAMETER(int32, bUseLightPowerDelta)
	SHADER_PARAMETER(FMatrix44f, UnjitteredClipToTranslatedWorld)
	SHADER_PARAMETER(FMatrix44f, UnjitteredTranslatedWorldToClip)
	SHADER_PARAMETER(FMatrix44f, UnjitteredPrevTranslatedWorldToClip)
	SHADER_PARAMETER_STRUCT_INCLUDE(FHZBParameters, HZBParameters)
	SHADER_PARAMETER(FIntPoint, VisibleLightHashViewMinInTiles)
	SHADER_PARAMETER(FIntPoint, VisibleLightHashViewSizeInTiles)
	SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, DownsampledSceneDepth)
	SHADER_PARAMETER_RDG_TEXTURE(Texture2D<UNORM float3>, DownsampledSceneWorldNormal)
	SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float>, LightPowerHistoryRatioBuffer)
END_SHADER_PARAMETER_STRUCT()

BEGIN_SHADER_PARAMETER_STRUCT(FMegaLightsVolumeParameters, )
	SHADER_PARAMETER(float, VolumeMinSampleWeight)
	SHADER_PARAMETER(int32, VolumeDownsampleFactorMultShift)
	SHADER_PARAMETER(int32, VolumeDebugSliceIndex)
	SHADER_PARAMETER(FIntVector, NumSamplesPerVoxel)
	SHADER_PARAMETER(FIntVector, NumSamplesPerVoxelDivideShift)
	SHADER_PARAMETER(FIntVector, DownsampledVolumeViewSize)
	SHADER_PARAMETER(FIntVector, VolumeViewSize)
	SHADER_PARAMETER(FIntVector, VolumeSampleViewSize)
	SHADER_PARAMETER(FVector3f, VolumeInvBufferSize)
	SHADER_PARAMETER(FVector3f, MegaLightsVolumeZParams)
	SHADER_PARAMETER(uint32, MegaLightsVolumePixelSize)
	SHADER_PARAMETER(FVector3f, VolumeFrameJitterOffset)
	SHADER_PARAMETER(float, VolumePhaseG)
	SHADER_PARAMETER(float, VolumeInverseSquaredLightDistanceBiasScale)
	SHADER_PARAMETER(float, LightSoftFading)
	SHADER_PARAMETER(uint32, TranslucencyVolumeCascadeIndex)
	SHADER_PARAMETER(float, TranslucencyVolumeInvResolution)
	SHADER_PARAMETER(uint32, UseHZBOcclusionTest)
	SHADER_PARAMETER(uint32, IsUnifiedVolume)
	SHADER_PARAMETER(FIntVector, ResampleVolumeViewSize)
	SHADER_PARAMETER(FVector3f, ResampleVolumeInvBufferSize)
	SHADER_PARAMETER(FVector3f, ResampleVolumeZParams)
END_SHADER_PARAMETER_STRUCT()

enum class EMegaLightsInput : uint8
{
	GBuffer,
	HairStrands,
	FrontLayerTranslucency,
	Count
};

enum class EMegaLightsDebugMode : uint8
{
	Disabled,
	GBuffer,
	Volume,
	TranslucencyVolume,
	HairStrands,
	FrontLayerTranslucency,

	MAX
};

enum class EMegaLightsDebugTarget : uint32
{
	None = 0,
	GBuffer = 0x1,
	HairStrands = 0x2,
	FrontLayerTranslucency = 0x4,
	Volume = 0x8,
	TranslucencyLightingVolume = 0x10,

	VolumeMask = Volume | TranslucencyLightingVolume
};
ENUM_CLASS_FLAGS(EMegaLightsDebugTarget)

// Internal functions, don't use outside of the MegaLights
namespace MegaLights
{
	// Various parameters passed to MegaLights visualizations done at the end of the frame after TSR
	BEGIN_SHADER_PARAMETER_STRUCT(FVisualizeParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, OpaqueOnlyDepthTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<UNORM float4>, VisualizeSpatialFilterTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<UNORM float4>, VisualizeSpatialFilterTexture2)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<UNORM float4>, VisualizeTemporalFilterTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, ShadingConfidenceTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint>, NumFramesAccumulatedTexture)
		SHADER_PARAMETER(float, TemporalMaxFramesAccumulated)
		SHADER_PARAMETER(float, HistorySpatialStdDevThreshold)
	END_SHADER_PARAMETER_STRUCT()

	BEGIN_SHADER_PARAMETER_STRUCT(FHairVoxelTraceParameters, )
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FHairStrandsViewUniformParameters, HairStrands)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FVirtualVoxelParameters, VirtualVoxel)
	END_SHADER_PARAMETER_STRUCT()

	void SetHairVoxelTraceParameters(const FViewInfo& View, FHairVoxelTraceParameters& PassParameters);

	// Common HWRT parameters
	BEGIN_SHADER_PARAMETER_STRUCT(FRayTracingParameters, )
		// Bias
		SHADER_PARAMETER(float, RayTracingBias)
		SHADER_PARAMETER(float, RayTracingNormalBias)
		SHADER_PARAMETER(float, RayTracingPullbackBias)
		SHADER_PARAMETER(float, HairNonShadowedLightMaxTraceDistance)

		// Distant Screen Traces
		SHADER_PARAMETER(float, DistantScreenTraceSlopeCompareTolerance)
		SHADER_PARAMETER(float, DistantScreenTraceStartDistance)
		SHADER_PARAMETER(float, DistantScreenTraceLength)

		// Visualize rays
		SHADER_PARAMETER(uint32, VisualizeRays)
		SHADER_PARAMETER(uint32, VisualizeRaysMinIterations)
		SHADER_PARAMETER(uint32, VisualizeRaysHeatmapMin)
		SHADER_PARAMETER(uint32, VisualizeRaysHeatmapMax)

		// FarField
		SHADER_PARAMETER(float, NearFieldSceneRadius)
		SHADER_PARAMETER(float, NearFieldMaxTraceDistance)
		SHADER_PARAMETER(float, NearFieldMaxTraceDistanceDitherScale)
		SHADER_PARAMETER(float, FarFieldBias)
		SHADER_PARAMETER(float, FarFieldMaxTraceDistance)

		// Config
		SHADER_PARAMETER(uint32, UseFarField)
		SHADER_PARAMETER(uint32, ForceTwoSided)
		SHADER_PARAMETER(uint32, MaxTraversalIterations)
		SHADER_PARAMETER(uint32, MeshSectionVisibilityTest)
		SHADER_PARAMETER(uint32, RayTracingSceneLightingChannelAndMask)

		SHADER_PARAMETER_RDG_BUFFER_SRV(RaytracingAccelerationStructure, TLAS)
		SHADER_PARAMETER_RDG_BUFFER_SRV(RaytracingAccelerationStructure, FarFieldTLAS)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer, RayTracingSceneMetadata)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWInstanceHitCountBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<Lumen::FHitGroupRootConstants>, HitGroupData)
		SHADER_PARAMETER_STRUCT_REF(FLumenHardwareRayTracingUniformBufferParameters, LumenHardwareRayTracingUniformBuffer)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FNaniteRayTracingUniformParameters, NaniteRayTracing)
	END_SHADER_PARAMETER_STRUCT()

	void SetRayTracingParameters(FRDGBuilder& GraphBuilder, const FViewInfo& View, FRayTracingParameters& PassParameters);

	BEGIN_SHADER_PARAMETER_STRUCT(FCompactedTraceParameters, )
		RDG_BUFFER_ACCESS(IndirectArgs, ERHIAccess::IndirectArgs | ERHIAccess::SRVCompute)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CompactedTraceTexelData)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CompactedTraceTexelAllocator)
	END_SHADER_PARAMETER_STRUCT()

	struct FTraceStats
	{
		FRDGBufferRef VSM = nullptr;
		FRDGBufferRef Screen = nullptr;
		FRDGBufferRef World = nullptr;
		FRDGBufferRef WorldMaterialRetrace = nullptr;
		FRDGBufferRef Volume = nullptr;
		FRDGBufferRef TranslucencyVolume[TVC_MAX] = {};
	};

	struct FVolumeCompactedTraceParameters
	{
		FCompactedTraceParameters Volume;
		FCompactedTraceParameters TranslucencyVolume[TVC_MAX];
	};

	struct FVolumeLightSampleParameters
	{
		FIntVector VolumeSampleBufferSize;
		FRDGTextureRef VolumeLightSamples;
		FRDGTextureRef VolumeLightSampleRays;
		FIntVector TranslucencyVolumeSampleBufferSize;
		TConstArrayView<FRDGTextureRef> TranslucencyVolumeLightSamples;
		TConstArrayView<FRDGTextureRef> TranslucencyVolumeLightSampleRays;
	};

	void RayTraceLightSamples(
		const FSceneViewFamily& ViewFamily,
		const FViewInfo& View, int32 ViewIndex,
		FRDGBuilder& GraphBuilder,
		const FSceneTextures& SceneTextures,
		const FVirtualShadowMapArray* VirtualShadowMapArray,
		const FBoxSphereBounds& FirstPersonWorldSpaceRepresentationViewBounds,
		const FIntPoint SampleBufferSize,
		FRDGTextureRef LightSamples,
		FRDGTextureRef LightSampleRays,
		const FMegaLightsParameters& MegaLightsParameters,
		const FMegaLightsVolumeParameters& MegaLightsVolumeParameters,
		const FMegaLightsVolumeParameters& MegaLightsTranslucencyVolumeParameters,
		const FVolumeLightSampleParameters* VolumeLightSampleParameters,
		const FTraceStats* VolumeTraceStats,
		EMegaLightsInput InputType,
		bool bDebug,
		ERDGPassFlags ComputePassFlags
	);

	FTraceStats RayTraceVolumeLightSamples(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		const FSceneViewFamily& ViewFamily,
		const FMegaLightsParameters& MegaLightsParameters,
		const FMegaLightsVolumeParameters& MegaLightsVolumeParameters,
		const FMegaLightsVolumeParameters& MegaLightsTranslucencyVolumeParameters,
		const FVolumeLightSampleParameters& LightSampleParameters,
		const FVolumeCompactedTraceParameters* CompactedTraceParameters,
		ERDGPassFlags ComputePassFlags);

	void MarkVSMPages(
		const FViewInfo& View, int32 ViewIndex,
		FRDGBuilder& GraphBuilder,
		const FVirtualShadowMapArray& VirtualShadowMapArray,
		const FIntPoint SampleBufferSize,
		FRDGTextureRef LightSamples,
		FRDGTextureRef LightSampleRays,
		const FMegaLightsParameters& MegaLightsParameters,
		EMegaLightsInput InputType);

	bool UseWaveOps(EShaderPlatform ShaderPlatform);
	EPixelFormat GetLightingDataFormat();

	FMegaLightsViewState::FResources& GetViewState(const FViewInfo& View, EMegaLightsInput InputType);

	EMegaLightsDebugMode GetDebugMode();
	bool IsDebugEnabled(EMegaLightsInput InputType);
	bool IsDebugEnabled(EMegaLightsDebugMode DebugMode);
	bool IsDebugEnabledForShadingPass(int32 ShadingPassIndex, EShaderPlatform InPlatform);
	int32 GetDebugTileClassificationMode();
	int32 GetVisualizeMode(const FViewInfo& View);
	EMegaLightsDebugTarget GetVisualizeLightComplexityTarget(const FViewInfo& View);
	bool ShouldShowLightSamplingCostHeatmap(const FViewInfo& View, EMegaLightsInput InputType);

	FIntPoint GetDownsampleFactorXY(EMegaLightsInput InputType, EShaderPlatform ShaderPlatform);
	FIntPoint GetNumSamplesPerPixel2d(EMegaLightsInput InputType);
	FIntPoint GetNumSamplesPerPixel2d(int32 NumSamplesPerPixel1d);
	FIntVector GetNumSamplesPerVoxel3d(int32 NumSamplesPerVoxel1d);

	bool UseFastClear(EMegaLightsInput InputType);
	bool SupportsSpatialFilter(EMegaLightsInput InputType);
	bool UseSoftShadows(EMegaLightsInput InputType);

	float GetTemporalHistoryDistanceThreshold();
	bool UseSpatialFilter(EMegaLightsInput InputType);
	bool UseTemporalFilter(EMegaLightsInput InputType);

	float GetTransmissionSampleWeight();

	void ModifyCompilationEnvironment(EShaderPlatform Platform, FShaderCompilerEnvironment& OutEnvironment);

	// Keep in sync with TILE_TYPE_* in shaders
	enum class ETileType : uint8
	{
		// Base
		SimpleShading = TILE_MODE_SIMPLE_SHADING,
		ComplexShading = TILE_MODE_COMPLEX_SHADING,
		SimpleShading_Rect = TILE_MODE_SIMPLE_SHADING_RECT,
		ComplexShading_Rect = TILE_MODE_COMPLEX_SHADING_RECT,
		SimpleShading_Rect_Textured = TILE_MODE_SIMPLE_SHADING_RECT_TEXTURED,
		ComplexShading_Rect_Textured = TILE_MODE_COMPLEX_SHADING_RECT_TEXTURED,
		Empty = TILE_MODE_EMPTY,

		// Substrate-Blendable
		SingleShading = TILE_MODE_SINGLE_SHADING,
		SingleShading_Rect = TILE_MODE_SINGLE_SHADING_RECT,
		SingleShading_Rect_Textured = TILE_MODE_SINGLE_SHADING_RECT_TEXTURED,

		// Substrate-Adaptive
		ComplexSpecialShading = TILE_MODE_COMPLEX_SPECIAL_SHADING,
		ComplexSpecialShading_Rect = TILE_MODE_COMPLEX_SPECIAL_SHADING_RECT,
		ComplexSpecialShading_Rect_Textured = TILE_MODE_COMPLEX_SPECIAL_SHADING_RECT_TEXTURED,

		SHADING_MAX_LEGACY = TILE_MODE_MAX_LEGACY,
		SHADING_MAX_SUBSTRATE_BLENDABLE = TILE_MODE_MAX_SUBSTRATE_BLENDABLE,
		SHADING_MAX_SUBSTRATE_ADAPTIVE = TILE_MODE_MAX_SUBSTRATE_ADAPTIVE
	};

	bool IsRectLightTileType(ETileType TileType);
	bool IsTexturedLightTileType(ETileType TileType);
	bool IsComplexTileType(ETileType TileType);
	TArray<int32> GetShadingTileTypes(EMegaLightsInput InputType, EShaderPlatform InPlatform);
	const TCHAR* GetTileTypeString(ETileType TileType);

	enum class EMaterialMode : uint8
	{
		Disabled,
		AHS,
		RetraceAHS,

		MAX
	};

	EMaterialMode GetMaterialMode();
};

namespace MegaLightsVolume
{
	uint32 GetDownsampleFactor(EShaderPlatform ShaderPlatform);
	FIntVector GetNumSamplesPerVoxel3d();
};

namespace MegaLightsTranslucencyVolume
{
	uint32 GetDownsampleFactor(EShaderPlatform ShaderPlatform, bool bUnifiedVolume);
	FIntVector GetNumSamplesPerVoxel3d(bool bUnifiedVolume);
};

BEGIN_SHADER_PARAMETER_STRUCT(FMegaLightsVolumeData, )
	SHADER_PARAMETER(FIntVector, ViewGridSizeInt)
	SHADER_PARAMETER(FVector3f, ViewGridSize)
	SHADER_PARAMETER(FIntVector, ResourceGridSizeInt)
	SHADER_PARAMETER(FVector3f, ResourceGridSize)
	SHADER_PARAMETER(FVector3f, GridZParams)
	SHADER_PARAMETER(FVector2f, SVPosToVolumeUV)
	SHADER_PARAMETER(FIntPoint, FogGridToPixelXY)
	SHADER_PARAMETER(float, MaxDistance)
END_SHADER_PARAMETER_STRUCT()

class FMegaLightsViewContext
{
public:
	FMegaLightsViewContext(
		FRDGBuilder& InGraphBuilder,
		const int32 InViewIndex,
		const FViewInfo& InView,
		const FSceneViewFamily& InViewFamily,
		const FScene* InScene,
		const FSceneTextures& InSceneTextures,
		bool bInUseVSM);

	void TileClassificationMark(uint32 ShadingPassIndex, ERDGPassFlags ComputePassFlags);

	// Dependency: HZB, light grid, light function atlas
	void SetupStageOne(
		const bool bInShouldRenderVolumetricFog,
		const bool bInShouldRenderTranslucencyVolume,
		TUniformBufferRef<FBlueNoise> BlueNoiseUniformBuffer,
		EMegaLightsInput InInputType);

	// Dependency: base pass, hairstrands base pass, stochastic lighting mark, lighting channel texture
	void SetupStageTwo(
		FRDGTextureRef LightingChannelsTexture,
		FLumenSceneFrameTemporaries& LumenFrameTemporaries,
		ERDGPassFlags ComputePassFlags);

	void GenerateSamples(
		FRDGTextureRef LightingChannelsTexture,
		uint32 ShadingPassIndex,
		ERDGPassFlags ComputePassFlags);

	void GenerateVolumeSamples(ERDGPassFlags ComputePassFlags);

	void MarkVSMPages(
		const FVirtualShadowMapArray& VirtualShadowMapArray);

	void RayTrace(
		const FVirtualShadowMapArray& VirtualShadowMapArray,
		const FBoxSphereBounds& FirstPersonWorldSpaceRepresentationViewBounds,
		uint32 ShadingPassIndex,
		ERDGPassFlags ComputePassFlags);

	void VolumeRayTrace(ERDGPassFlags ComputePassFlags);

	void Resolve(
		FRDGTextureRef OutputColorTarget,
		FMegaLightsVolume* MegaLightsVolume,
		uint32 ShadingPassIndex,
		ERDGPassFlags ComputePassFlags);

	FMegaLightsVolume VolumeResolve(ERDGPassFlags ComputePassFlags);

	void DenoiseLighting(FRDGTextureRef OutputColorTarget, ERDGPassFlags ComputePassFlags);

	bool HasValidLightSamplingCostEstimate() const;
	void Visualize(FScreenPassTexture Output, int32 VisualizeMode, int32 VisualizeTileIndex, uint32 NumOverviewTilesPerRow);
	void VisualizeLightComplexity(const FScreenPassRenderTarget& OutputSceneColor, const FScreenPassTexture& InputSceneColor);

	void DispatchDebugTileClassificationPasses(ERDGPassFlags ComputePassFlags);
	void DispatchVisualizeLightsPasses(ERDGPassFlags ComputePassFlags);

	bool AreSamplesGenerated() const
	{
		return bSamplesGenerated;
	}

	bool AreVolumeSamplesGenerated() const
	{
		return bVolumeSamplesGenerated;
	}

	bool VolumeUseAsyncCompute() const;

	bool GenerateSamplesUseAsyncCompute() const;

	bool ShouldRenderTranslucencyVolume() const
	{
		return bShouldRenderTranslucencyVolume;
	}

	uint32 GetReferenceShadingPassCount() const { return ReferenceShadingPassCount; }

	void SetFrontLayerTranslucencyData(const FFrontLayerTranslucencyData& InFrontLayerTranslucencyData);
	bool HasFrontLayerTranslucencyData() const;

	void UpdateHZB();

private:
	bool IsAsyncComputeAllowed() const;

	void InitVolumeVisibleLightHash(ERDGPassFlags ComputePassFlags);
	void BuildVolumeVisibleLightHash(ERDGPassFlags ComputePassFlags);
	void FilterVolumeVisibleLightHash(ERDGPassFlags ComputePassFlags);
	FMegaLightsVolume ShadeVolumeLightSamples(uint32 ShadingPassIndex, ERDGPassFlags ComputePassFlags);

	FRDGBuilder& GraphBuilder;
	const int32 ViewIndex;
	const FViewInfo& View;
	const FSceneViewFamily& ViewFamily;
	const FScene* Scene;
	const FSceneTextures& SceneTextures;
	const bool bUseVSM;

	const FFrontLayerTranslucencyData* FrontLayerTranslucencyData = nullptr;

	bool bSamplesGenerated = false;
	bool bVolumeSamplesGenerated = false;
	bool bVolumeRaysTraced = false;
	bool bVolumeLightingResolved = false;

	EMegaLightsInput InputType;

	bool bUnifiedVolume;
	bool bVolumeEnabled;
	bool bGuideByHistory = true;
	bool bVolumeGuideByHistory;
	bool bTranslucencyVolumeGuideByHistory;
	bool bDebug = false;
	bool bVisualizeLightComplexityFrozen = false;
	bool bVisualizeLightComplexityDump = false;
	bool bVolumeDebug;
	bool bTranslucencyVolumeDebug;
	bool bUseLightFunctionAtlas;
	bool bSpatial;
	bool bTemporal;
	bool bSubPixelShading;
	bool bShouldRenderVolumetricFog;
	bool bShouldRenderTranslucencyVolume;
	bool bUseHairComplexTransmittance = false;
	bool bUseLightPowerDelta = false;

	int32 DebugTileClassificationMode = 0;
	EMegaLightsDebugTarget VisualizeLightComplexityTarget = EMegaLightsDebugTarget::None;

	FMegaLightsParameters MegaLightsParameters;
	FMegaLightsVolumeParameters MegaLightsVolumeParameters;
	FMegaLightsVolumeParameters MegaLightsTranslucencyVolumeParameters;

	FMegaLightsVolumeData VolumeParameters;
	FVolumetricFogGlobalData VolumetricFogParamaters;

	FIntPoint DownsampleFactor;
	FIntPoint SampleBufferSize;
	FIntPoint DonwnsampledSampleBufferSize;

	FIntPoint NumSamplesPerPixel2d;
	FIntVector NumSamplesPerVoxel3d;
	FIntVector NumSamplesPerTranslucencyVoxel3d;

	FIntPoint ViewSizeInTiles;
	FIntPoint DownsampledViewSizeInTiles;

	uint32 VisibleLightHashBufferSize = 0;
	uint32 HiddenLightHashBufferSize = 0;
	FIntPoint VisibleLightHashSizeInTiles;
	FIntPoint VisibleLightHashViewMinInTiles;
	FIntPoint VisibleLightHashViewSizeInTiles;

	uint32 VolumeDownsampleFactor;
	FIntVector VolumeBufferSize;
	FIntVector VolumeSampleBufferSize;
	FIntVector VolumeViewSize;

	FRDGTextureRef VolumeLightSamples = nullptr;
	FRDGTextureRef VolumeLightSampleRays = nullptr;

	uint32 VolumeVisibleLightHashBufferSize = 0;
	uint32 VolumeHiddenLightHashBufferSize = 0;
	FIntVector VolumeVisibleLightHashTileSize;
	FIntVector VolumeVisibleLightHashSizeInTiles;
	FIntVector VolumeVisibleLightHashViewSizeInTiles;
	FIntVector VolumeDownsampledViewSize;

	uint32 TranslucencyVolumeDownsampleFactor;
	FIntVector TranslucencyVolumeBufferSize;
	FIntVector TranslucencyVolumeSampleBufferSize;
	FIntVector TranslucencyVolumeDownsampledBufferSize;

	TArray<FRDGTextureRef, TInlineAllocator<TVC_MAX>> TranslucencyVolumeLightSamples;
	TArray<FRDGTextureRef, TInlineAllocator<TVC_MAX>> TranslucencyVolumeLightSampleRays;

	uint32 TranslucencyVolumeVisibleLightHashBufferSize = 0;
	uint32 TranslucencyVolumeHiddenLightHashBufferSize = 0;
	FIntVector TranslucencyVolumeVisibleLightHashTileSize;
	FIntVector TranslucencyVolumeVisibleLightHashSizeInTiles;

	FRDGTextureRef SceneDepth = nullptr;
	FRDGTextureRef SceneWorldNormal = nullptr;
	FRDGTextureRef DownsampledSceneDepth = nullptr;
	FRDGTextureRef DownsampledSceneWorldNormal = nullptr;

	FRDGBufferRef TileIndirectArgs = nullptr;
	FRDGBufferRef TileAllocator = nullptr;
	FRDGBufferRef TileData = nullptr;
	FRDGBufferRef DownsampledTileIndirectArgs = nullptr;
	FRDGBufferRef DownsampledTileAllocator = nullptr;
	FRDGBufferRef DownsampledTileData = nullptr;

	FRDGTextureRef LightSamples = nullptr;
	FRDGTextureRef LightSampleRays = nullptr;

	TArray<int32> ShadingTileTypes;

	StochasticLighting::FHistoryScreenParameters HistoryScreenParameters;
	FIntPoint HistoryVisibleLightHashViewMinInTiles = 0;
	FIntPoint HistoryVisibleLightHashViewSizeInTiles = 0;
	FRDGTextureRef DiffuseLightingHistory = nullptr;
	FRDGTextureRef SpecularLightingHistory = nullptr;
	FRDGTextureRef LightingMomentsHistory = nullptr;
	FRDGTextureRef SceneDepthHistory = nullptr;
	FRDGTextureRef SceneNormalAndShadingHistory = nullptr;
	FRDGTextureRef NumFramesAccumulatedHistory = nullptr;
	FRDGBufferRef VisibleLightHashHistory = nullptr;
	FRDGBufferRef HiddenLightHashHistory = nullptr;

	FIntVector HistoryVolumeVisibleLightHashViewSizeInTiles = FIntVector::ZeroValue;
	FRDGBufferRef VolumeVisibleLightHashHistory = nullptr;
	FRDGBufferRef VolumeHiddenLightHashHistory = nullptr;

	FIntVector HistoryTranslucencyVolumeVisibleLightHashSizeInTiles = FIntVector::ZeroValue;
	FRDGBufferRef TranslucencyVolumeVisibleLightHashHistory[TVC_MAX] = {};
	FRDGBufferRef TranslucencyVolumeHiddenLightHashHistory[TVC_MAX] = {};

	// Visualization parameters
	FRDGTextureRef LightSamplingCostEstimate = nullptr;
	MegaLights::FVisualizeParameters MegaLightsVisualizeParameters;

	// State for the shading loop; much of this gets lazily created in the loop
	// This should perhaps be moved to a separate context structure in the future
	uint32 ReferenceShadingPassCount;
	bool bReferenceMode;
	uint32 FirstPassStateFrameIndex;
	EPixelFormat AccumulatedRGBLightingDataFormat;
	EPixelFormat AccumulatedRGBALightingDataFormat;
	EPixelFormat AccumulatedConfidenceDataFormat;

	FRDGTextureRef ResolvedDiffuseLighting = nullptr;
	FRDGTextureRef ResolvedSpecularLighting = nullptr;
	FRDGTextureRef TempDiffuseSnapshot = nullptr;
	FRDGTextureRef TempSpecularSnapshot = nullptr;
	FRDGTextureRef ShadingConfidence = nullptr;
	FRDGTextureRef VolumeResolvedLighting = nullptr;
	FRDGBufferRef VisibleLightHash = nullptr;
	FRDGBufferRef HiddenLightHash = nullptr;
	FRDGBufferRef VolumeVisibleLightHash = nullptr;
	FRDGBufferRef VolumeHiddenLightHash = nullptr;
	FRDGTextureRef TranslucencyVolumeResolvedLightingAmbient[TVC_MAX] = {};
	FRDGTextureRef TranslucencyVolumeResolvedLightingDirectional[TVC_MAX] = {};

	FRDGBufferRef TranslucencyVolumeVisibleLightHash[TVC_MAX] = {};
	FRDGBufferRef TranslucencyVolumeHiddenLightHash[TVC_MAX] = {};

	MegaLights::FTraceStats VolumeTraceStats;
};

class FGPUBufferReadbackCollection
{
public:
	FGPUBufferReadbackCollection();
	~FGPUBufferReadbackCollection();

	bool EnqueueReadback(FRDGBuilder& GraphBuilder, FRDGBufferRef BufferToReadback, uint32 SizeInBytes);

	void* LockReadback();
	void UnlockReadback();

private:
	static const int32 NumReadbacks = 3;

	int32 WritePtr;
	int32 NumPending;
	uint32 ReadbackSizeArray[NumReadbacks];
	FRHIGPUBufferReadback* Readbacks[NumReadbacks];
	FRHIGPUBufferReadback* LockedReadback;
};

// Temporaries valid only in a single frame
struct FMegaLightsFrameTemporaries
{
	TArray<FMegaLightsViewContext, SceneRenderingAllocator> ViewContexts;
	TArray<FMegaLightsViewContext, SceneRenderingAllocator> ViewContextsHairStrands;
	TArray<FMegaLightsViewContext, SceneRenderingAllocator> ViewContextsFrontLayerTranslucency;
	TUniformBufferRef<FBlueNoise> BlueNoiseUniformBuffer;

	void UpdateHZB();
};
