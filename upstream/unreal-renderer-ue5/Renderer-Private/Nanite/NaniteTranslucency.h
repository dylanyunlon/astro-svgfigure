// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "ShaderParameters.h"
#include "VertexFactory.h"
#include "GlobalRenderResources.h"
#include "NaniteDefinitions.h"
#include "NaniteShared.h"
#include "TranslucentPassResource.h"
#include "BasePassRendering.h"
#include "MeshPassProcessor.h"

// Vertex factory that performs Nanite translucent material shading.
class FNaniteTranslucencyFactory final : public FVertexFactory
{
	DECLARE_VERTEX_FACTORY_TYPE(FNaniteTranslucencyFactory);

public:
	FNaniteTranslucencyFactory(ERHIFeatureLevel::Type FeatureLevel);
	~FNaniteTranslucencyFactory();

	virtual void InitRHI(FRHICommandListBase& RHICmdList) override final;

	static bool ShouldCompilePermutation(const FVertexFactoryShaderPermutationParameters& Parameters);
	static void ModifyCompilationEnvironment(const FVertexFactoryShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment);
};

class FViewInfo;
class FScreenPassTextureViewport;
class FNaniteTranslucencyParameters;
class FTranslucentBasePassParameters;

namespace Nanite
{

struct FRasterResults;

void SetTranslucencyParameters(
	FRDGBuilder& GraphBuilder,
	const FScene* Scene,
	const FViewInfo& View,
	const Nanite::FRasterResults* RasterResults,
	FNaniteTranslucencyParameters& Parameters
);

void RenderTranslucency(
	FRDGBuilder& GraphBuilder,
	const FSceneRenderer& SceneRenderer,
	FViewInfo& View, int32 ViewIndex,
	FScreenPassTextureViewport Viewport,
	float ViewportScale,
	FRDGTextureMSAA SceneColorTexture,
	ERenderTargetLoadAction SceneColorLoadAction,
	FRDGTextureRef SceneDepthTexture,
	FTranslucentBasePassParameters* PassParameters,
	TRDGUniformBufferRef<FTranslucentBasePassUniformParameters> BasePassParameters,
	ETranslucencyPass::Type TranslucencyPass,
	FRDGBufferRef RasterBinArgs,
	const FNaniteTranslucencyContext& TranslucencyContext
);

class FTranslucencyFactoryResource : public FRenderResource
{
public:
	virtual void InitRHI(FRHICommandListBase& RHICmdList) override;
	virtual void ReleaseRHI() override;

	FNaniteTranslucencyFactory* GetVertexFactory() { return VertexFactory; }

private:
	FNaniteTranslucencyFactory* VertexFactory = nullptr;
};

extern TGlobalResource<FTranslucencyFactoryResource> GTranslucencyFactoryResource;

}

struct FNaniteTranslucencyPassData
{
	TShaderRef<TBasePassMeshShaderPolicyParamType<FUniformLightMapPolicy>> TypedMeshShader;
	TShaderRef<TBasePassVertexShaderPolicyParamType<FUniformLightMapPolicy>> TypedVertexShader;
	TShaderRef<TBasePassPixelShaderPolicyParamType<FUniformLightMapPolicy>> TypedPixelShader;
	const FMaterialRenderProxy* MaterialProxy = nullptr;
	const FMaterial* Material = nullptr;
	FMeshDrawShaderBindings ShaderBindingsMSPS;
	FMeshDrawShaderBindings ShaderBindingsVSPS;
};
