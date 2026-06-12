// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "RenderGraphFwd.h"
#include "SceneTexturesConfig.h"
#include "ScreenPass.h"
#include "ShaderParameterMacros.h"
#include "Math/IntPoint.h"
#include "LumenReflections.h"
#include "ShaderPrintParameters.h"
#include "StochasticLighting/StochasticLightingVisualize.h"

class FScene;
class FSceneTextureParameters;
class FViewInfo;
class FLumenCardTracingParameters;
class FLumenIndirectTracingParameters;
struct FLumenSceneFrameTemporaries;
struct FEngineShowFlags;

// r.Lumen.Visualize.Mode
#define VISUALIZE_MODE_OVERVIEW						1
#define VISUALIZE_MODE_PERFORMANCE_OVERVIEW			2
#define VISUALIZE_MODE_LUMEN_SCENE					3
#define VISUALIZE_MODE_REFLECTION_VIEW				4
#define VISUALIZE_MODE_SURFACE_CACHE				5
#define VISUALIZE_MODE_GEOMETRY_NORMALS				6
#define VISUALIZE_MODE_DEDICATED_REFLECTION_RAYS	7
#define VISUALIZE_MODE_ALBEDO						8
#define VISUALIZE_MODE_NORMALS						9
#define VISUALIZE_MODE_CARD_SHARING_ID				21
#define VISUALIZE_MODE_SCREENPROBEGATHER_FAST_UPDATE_MODE_AMOUNT	22
#define VISUALIZE_MODE_SCREENPROBEGATHER_NUM_FRAMES_ACCUMULATED		23
#define VISUALIZE_MODE_RADIOSITY_NUM_FRAMES_ACCUMULATED				24

namespace LumenVisualize
{
	BEGIN_SHADER_PARAMETER_STRUCT(FSceneParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(StochasticLightingVisualize::FTonemappingParameters, TonemappingParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrintUniformBuffer)
		SHADER_PARAMETER(FIntPoint, InputViewSize)
		SHADER_PARAMETER(FIntPoint, InputViewOffset)
		SHADER_PARAMETER(FIntPoint, OutputViewSize)
		SHADER_PARAMETER(FIntPoint, OutputViewOffset)
		SHADER_PARAMETER(int32, VisualizeHiResSurface)
		SHADER_PARAMETER(int32, VisualizeMode)
		SHADER_PARAMETER(uint32, VisualizeCullingMode)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenReflections::FCompositeParameters, ReflectionsCompositeParameters)
		SHADER_PARAMETER_TEXTURE(Texture2D, PreIntegratedGF)
		SHADER_PARAMETER_SAMPLER(SamplerState, PreIntegratedGFSampler)
		SHADER_PARAMETER(uint32, MaxReflectionBounces)
		SHADER_PARAMETER(uint32, MaxRefractionBounces)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, MeshCardsIndexToCardSharingIdBuffer)
	END_SHADER_PARAMETER_STRUCT()

	void VisualizeHardwareRayTracing(
		FRDGBuilder& GraphBuilder,
		const FScene* Scene,
		const FSceneTextureParameters& SceneTextures,
		const FViewInfo& View,
		const FLumenSceneFrameTemporaries& FrameTemporaries,
		const FLumenCardTracingParameters& TracingParameters,
		FLumenIndirectTracingParameters& IndirectTracingParameters,
		LumenVisualize::FSceneParameters& VisualizeParameters,
		FRDGTextureRef SceneColor,
		bool bVisualizeModeWithHitLighting,
		EDiffuseIndirectMethod DiffuseIndirectMethod);

	bool IsHitLightingForceEnabled(const FViewInfo& View, EDiffuseIndirectMethod DiffuseIndirectMethod);
	bool UseHitLighting(const FViewInfo& View, EDiffuseIndirectMethod DiffuseIndirectMethod);
	bool UseSurfaceCacheFeedback(const FEngineShowFlags& ShowFlags);
};

struct FVisualizeLumenSceneInputs
{
	// [Optional] Render to the specified output. If invalid, a new texture is created and returned.
	FScreenPassRenderTarget OverrideOutput;

	// [Required] The scene color
	FScreenPassTexture SceneColor;

	// [Required] The scene depth
	FScreenPassTexture SceneDepth;

	// [Required] Used when scene textures are required by the material.
	FSceneTextureShaderParameters SceneTextures;
};

extern FScreenPassTexture AddVisualizeLumenScenePass(FRDGBuilder& GraphBuilder, const FViewInfo& View, EDiffuseIndirectMethod DiffuseIndirectMethod, EReflectionsMethod ReflectionsMethod, const FVisualizeLumenSceneInputs& Inputs, const FLumenSceneFrameTemporaries& FrameTemporaries);

extern int32 GetLumenVisualizeMode(const FViewInfo& View);