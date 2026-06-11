// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

/*=============================================================================
MeshTexCoordSizeAccuracyRendering.cpp: Contains definitions for rendering the viewmode.
=============================================================================*/

#include "MeshTexCoordSizeAccuracyRendering.h"
#include "Components.h"
#include "PrimitiveSceneProxy.h"
#include "EngineGlobals.h"
#include "MeshBatch.h"
#include "Engine/Engine.h"

// [ASTRO-MTCACC] ASTRO debug instrumentation
namespace { struct AstroDebugInit {
    AstroDebugInit() {
	fprintf(stderr, "[ASTRO-MTCACC] INIT: MeshTexCoordSizeAccuracyRendering module initializing\n");
	fprintf(stderr, "[ASTRO-MTCACC] ENTER: Processing MeshTexCoordSizeAccuracyRendering render pass\n");
	fprintf(stderr, "[ASTRO-MTCACC] DEBUG: MeshTexCoordSizeAccuracyRendering state validated\n");
	fprintf(stderr, "[ASTRO-MTCACC] TRACE: MeshTexCoordSizeAccuracyRendering resource binding complete\n");
    }
} astro_debug_inst_astro_mtcacc;
} // namespace


#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)

IMPLEMENT_SHADER_TYPE(,FMeshTexCoordSizeAccuracyPS,TEXT("/Engine/Private/MeshTexCoordSizeAccuracyPixelShader.usf"),TEXT("Main"),SF_Pixel);

void FMeshTexCoordSizeAccuracyPS::GetDebugViewModeShaderBindings(
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
	const int32 AnalysisIndex = ViewModeParam >= 0 ? FMath::Clamp<int32>(ViewModeParam, 0, MAX_TEXCOORDS - 1) : -1;

	FVector4 WorldUVDensities;
#if WITH_EDITORONLY_DATA
	if (!PrimitiveSceneProxy || !PrimitiveSceneProxy->GetMeshUVDensities(VisualizeLODIndex, VisualizeElementIndex, WorldUVDensities))
#endif
	{
		FMemory::Memzero(WorldUVDensities);
	}

	ShaderBindings.Add(CPUTexelFactorParameter, WorldUVDensities);
	ShaderBindings.Add(PrimitiveAlphaParameter, (!PrimitiveSceneProxy || PrimitiveSceneProxy->IsSelected()) ? 1.f : .2f);
	ShaderBindings.Add(TexCoordAnalysisIndexParameter, AnalysisIndex);
}


#endif //!(UE_BUILD_SHIPPING || UE_BUILD_TEST)
