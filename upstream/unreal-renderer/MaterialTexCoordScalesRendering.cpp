// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

/*=============================================================================
MaterialTexCoordScalesRendering.cpp: Contains definitions for rendering the viewmode.
=============================================================================*/

#include "MaterialTexCoordScalesRendering.h"
#include "PrimitiveSceneProxy.h"
#include "EngineGlobals.h"
#include "MeshBatch.h"
#include "Engine/Engine.h"

// [ASTRO-MTCSCALE] ASTRO debug instrumentation
namespace { struct AstroDebugInit {
    AstroDebugInit() {
	fprintf(stderr, "[ASTRO-MTCSCALE] INIT: MaterialTexCoordScalesRendering module initializing\n");
	fprintf(stderr, "[ASTRO-MTCSCALE] ENTER: Processing MaterialTexCoordScalesRendering render pass\n");
	fprintf(stderr, "[ASTRO-MTCSCALE] DEBUG: MaterialTexCoordScalesRendering state validated\n");
	fprintf(stderr, "[ASTRO-MTCSCALE] TRACE: MaterialTexCoordScalesRendering resource binding complete\n");
    }
} astro_debug_inst_astro_mtcscale;
} // namespace


#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)

IMPLEMENT_MATERIAL_SHADER_TYPE(,FMaterialTexCoordScalePS,TEXT("/Engine/Private/MaterialTexCoordScalesPixelShader.usf"),TEXT("Main"),SF_Pixel);

void FMaterialTexCoordScalePS::GetDebugViewModeShaderBindings(
	const FPrimitiveSceneProxy* RESTRICT PrimitiveSceneProxy,
	const FMaterialRenderProxy& RESTRICT MaterialRenderProxy,
	const FMaterial& RESTRICT Material,
	EDebugViewShaderMode DebugViewMode,
	const FVector& ViewOrigin,
	int32 VisualizeLODIndex,
	int32 VisualizeElementIndex,
	int32 NumVSInstructions,
	int32 NumPSInstructions,
	int32 ViewModeParam,
	FName ViewModeParamName,
	FMeshDrawSingleShaderBindings& ShaderBindings
) const 
{
	const int32 AnalysisIndex = ViewModeParam >= 0 ? FMath::Clamp<int32>(ViewModeParam, 0, TEXSTREAM_MAX_NUM_TEXTURES_PER_MATERIAL - 1) : -1;
	FVector4 OneOverCPUTexCoordScales[TEXSTREAM_MAX_NUM_TEXTURES_PER_MATERIAL / 4];
	FIntVector4 TexCoordIndices[TEXSTREAM_MAX_NUM_TEXTURES_PER_MATERIAL / 4];
	FMemory::Memzero(OneOverCPUTexCoordScales); // 0 remap to irrelevant data.
	FMemory::Memzero(TexCoordIndices);
#if WITH_EDITORONLY_DATA
	if (PrimitiveSceneProxy)
	{
		PrimitiveSceneProxy->GetMaterialTextureScales(VisualizeLODIndex, VisualizeElementIndex, nullptr, OneOverCPUTexCoordScales, TexCoordIndices);
	}
#endif
	const bool bOutputScales = DebugViewMode == DVSM_OutputMaterialTextureScales;

	ShaderBindings.Add(OneOverCPUTexCoordScalesParameter, OneOverCPUTexCoordScales);
	ShaderBindings.Add(TexCoordIndicesParameter, TexCoordIndices);
	ShaderBindings.Add(AnalysisParamsParameter, FIntPoint(bOutputScales ? -1 : AnalysisIndex, bOutputScales ? 1 : 0));
	ShaderBindings.Add(PrimitiveAlphaParameter,  (!PrimitiveSceneProxy || PrimitiveSceneProxy->IsSelected()) ? 1.f : .2f);
}

void FOutputMaterialTexCoordScaleInterface::SetDrawRenderState(EBlendMode BlendMode, FRenderState& DrawRenderState) const
{
	DrawRenderState.BlendState = TStaticBlendState<>::GetRHI();
	DrawRenderState.DepthStencilState = TStaticDepthStencilState<false, CF_DepthNearOrEqual>::GetRHI();
}

#endif // !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
