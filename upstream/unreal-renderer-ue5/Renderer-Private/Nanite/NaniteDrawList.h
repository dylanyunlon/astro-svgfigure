// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "NaniteShared.h"
#include "NaniteMaterials.h"
#include "PrimitiveSceneInfo.h"

class FNaniteMaterialListContext
{
public:

	struct FMaterialSectionData
	{
		FNaniteRasterPipeline RasterPipeline;
		FNaniteShadingPipeline TriangleShadingPipeline;
		FNaniteShadingPipeline VoxelShadingPipeline;
		FNaniteShadingPipeline CurveShadingPipeline;
		FGraphEventArray RasterAsyncPSOCompilationEvents;
		FGraphEventArray TriangleShadingAsyncPSOCompilationEvents;
	};

	struct FDeferredPipelines
	{
		FPrimitiveSceneInfo* PrimitiveSceneInfo;
		TArray<FMaterialSectionData, TInlineAllocator<4>> MaterialSectionData;
	};

public:
	void Apply(FScene& Scene);

private:
	FNaniteMaterialSlot& GetMaterialSlotForWrite(FPrimitiveSceneInfo& PrimitiveSceneInfo, ENaniteMeshPass::Type MeshPass, uint8 SectionIndex);
	void AddRasterBin(FPrimitiveSceneInfo& PrimitiveSceneInfo, const FNaniteRasterBin& PrimaryRasterBin, const FNaniteRasterBin& FallbackRasterBin, ENaniteMeshPass::Type MeshPass, uint8 SectionIndex);
	void AddShadingBin(FPrimitiveSceneInfo& PrimitiveSceneInfo, const FNaniteShadingBin& TriangleShadingBin, const FNaniteShadingBin& VoxelShadingBin, const FNaniteShadingBin& CurveShadingBin, ENaniteMeshPass::Type MeshPass, uint8 SectionIndex);

public:
	TArray<FDeferredPipelines> DeferredPipelines[ENaniteMeshPass::Num];
};
