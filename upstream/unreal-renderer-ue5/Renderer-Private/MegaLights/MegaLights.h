// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "RenderGraphFwd.h"
#include "SceneView.h"
#include "ScreenPass.h"

namespace EMegaLightsShadowMethod
{
	enum Type : int;
}

class FSceneViewFamily;
class FViewInfo;
struct FGlobalShaderPermutationParameters;
struct FLumenSceneFrameTemporaries;
struct FMegaLightsFrameTemporaries;
enum EShaderPlatform : uint16;
class FRDGTexture;
using FRDGTextureRef = FRDGTexture*;

struct FScreenMessageWriter;

namespace ECastRayTracedShadow
{
	enum Type : int;
};

namespace StochasticLighting
{
	enum class EMaterialSource;
}

class FMegaLightsVolume
{
public:
	FRDGTextureRef Texture = nullptr;
	FRDGTextureRef TranslucencyAmbient[TVC_MAX] = {};
	FRDGTextureRef TranslucencyDirectional[TVC_MAX] = {};
};

enum class EMegaLightsMode
{
	Disabled,
	EnabledRT,
	EnabledVSM
};

// Public MegaLights interface
namespace MegaLights
{
	bool IsEnabled(const FSceneViewFamily& ViewFamily);
	bool IsEnabledInProject(EShaderPlatform ShaderPlatform);

	bool IsUsingClosestHZB(const FSceneViewFamily& ViewFamily);
	bool IsUsingGlobalSDF(const FSceneViewFamily& ViewFamily);
	bool IsUsingLightFunctions(const FSceneViewFamily& ViewFamily);
	bool IsUsingLightingChannels();

	bool IsSoftwareRayTracingSupported(const FSceneViewFamily& ViewFamily);
	bool IsHardwareRayTracingSupported(const FSceneViewFamily& ViewFamily);

	EMegaLightsMode GetMegaLightsMode(const FSceneViewFamily& ViewFamily, uint8 LightType, bool bLightAllowsMegaLights, TEnumAsByte<EMegaLightsShadowMethod::Type> ShadowMethod);
	bool UseHardwareRayTracing(const FSceneViewFamily& ViewFamily);
	bool UseInlineHardwareRayTracing(const FSceneViewFamily& ViewFamily);
	bool ShouldCompileShaders(EShaderPlatform ShaderPlatform);
	bool ShouldCompileShadersForReferenceMode(EShaderPlatform ShaderPlatform, bool HasEditorOnlyData);

	bool UseHairVoxelTraces(const FViewInfo& View);
	bool UseFarField(const FSceneViewFamily& ViewFamily);

	bool UseVolume();

	bool UseTranslucencyVolume();
	bool UseTranslucencyVolumeMarkTexture(const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries);
	bool IsTranslucencyVolumeSpatialFilterEnabled();
	bool IsTranslucencyVolumeTemporalFilterEnabled();

	bool UseFrontLayerTranslucencyDirectLighting(const FViewInfo& View);

	bool UseLightPowerDelta(const FViewInfo& View);
	EPixelFormat GetLightPowerDataFormat();

	bool IsHairStrandsEnabled(const FViewInfo& View);

	bool IsMarkingVSMPages();

	bool VolumeUseAsyncCompute(const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries);
	bool GenerateSamplesUseAsyncCompute(const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries);

	bool ShouldAddVisualizePostProcessingPass(const FViewInfo& View);
	bool ShouldAddVisualizePostProcessingPass(const FViewInfo& View, int32 ViewIndex, const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries, bool bOverview);
	bool IsUsingLightNames(const FViewInfo& View);
	bool IsVisualizeLightComplexityFrozen();
	bool ShouldAddVisualizeLightComplexityPostProcessingPass(const FViewInfo& View, int32 ViewIndex, const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries);
	FScreenPassTexture AddVisualizeLightComplexityPostProcessingPass(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		int32 ViewIndex,
		const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries,
		FScreenPassTexture InputSceneColor,
		FScreenPassRenderTarget OverrideOutput);
	bool IsVisualizeRaysEnabled();

	uint32 GetSampleMargin();
	float GetMinSampleWeightEstimate();

	bool HasWarning(const FViewInfo& View);
	void WriteWarnings(const FViewInfo& View, FScreenMessageWriter& Writer);

	BEGIN_SHADER_PARAMETER_STRUCT(FTileClassifyParameters, )
		SHADER_PARAMETER(float, MegaLightsHistoryDistanceThreshold)
		SHADER_PARAMETER(uint32, EnableTexturedRectLights)
	END_SHADER_PARAMETER_STRUCT()

	void SetupTileClassifyParameters(const FViewInfo& View, MegaLights::FTileClassifyParameters& OutParameters);

	FIntPoint GetDownsampleFactorXY(StochasticLighting::EMaterialSource MaterialSource, EShaderPlatform ShaderPlatform);
};