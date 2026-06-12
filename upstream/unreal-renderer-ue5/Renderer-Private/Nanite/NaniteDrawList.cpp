// Copyright Epic Games, Inc. All Rights Reserved.

#include "NaniteDrawList.h"
#include "BasePassRendering.h"
#include "NaniteSceneProxy.h"
#include "NaniteShading.h"
#include "NaniteVertexFactory.h"
#include "SceneUtils.h"
#include "ScenePrivate.h"
#include "MeshPassProcessor.inl"
#include "NaniteMaterialsSceneExtension.h"

int32 GNaniteAllowProgrammableDistances = 1;
static FAutoConsoleVariableRef CVarNaniteAllowProgrammableDistances(
	TEXT("r.Nanite.AllowProgrammableDistances"),
	GNaniteAllowProgrammableDistances,
	TEXT("Whether or not to allow disabling of Nanite programmable raster features (World Position Offset, Pixel Depth Offset, ")
	TEXT("Masked Opaque, or Displacement) at a distance from the camera."),
	ECVF_ReadOnly
);

static const TCHAR* NaniteRasterPSOCollectorName = TEXT("NaniteRaster");
static const TCHAR* NaniteShadingPSOCollectorName = TEXT("NaniteShading");
static const TCHAR* NaniteLumenCardPSOCollectorName = TEXT("NaniteLumenCard");

class FNaniteBasePSOCollector : public IPSOCollector
{
public:
	FNaniteBasePSOCollector(const TCHAR* NanitePSOCollectorName, ERHIFeatureLevel::Type InFeatureLevel) :
		IPSOCollector(FPSOCollectorCreateManager::GetIndex(GetFeatureLevelShadingPath(InFeatureLevel), NanitePSOCollectorName)),
		FeatureLevel(InFeatureLevel)
	{
	}

	virtual void CollectPSOInitializers(
		const FSceneTexturesConfig& SceneTexturesConfig,
		const FMaterial& Material,
		const FPSOPrecacheVertexFactoryData& VertexFactoryData,
		const FPSOPrecacheParams& PreCacheParams,
		TArray<FPSOPrecacheData>& PSOInitializers
	) override final;

protected:

	virtual void CollectNanitePSOInitializers(
		const FSceneTexturesConfig& SceneTexturesConfig,
		const FPSOPrecacheVertexFactoryData& VertexFactoryData,
		const FMaterial& Material,
		const FPSOPrecacheParams& PreCacheParams,
		EShaderPlatform ShaderPlatform,
		TArray<FPSOPrecacheData>& PSOInitializers) = 0;

	ERHIFeatureLevel::Type FeatureLevel;
};

void FNaniteBasePSOCollector::CollectPSOInitializers(
	const FSceneTexturesConfig& SceneTexturesConfig,
	const FMaterial& Material,
	const FPSOPrecacheVertexFactoryData& VertexFactoryData,
	const FPSOPrecacheParams& PreCacheParams,
	TArray<FPSOPrecacheData>& PSOInitializers
)
{
	// Make sure Nanite rendering is supported.
	EShaderPlatform ShaderPlatform = GetFeatureLevelShaderPlatform(FeatureLevel);
	if (!UseNanite(ShaderPlatform))
	{
		return;
	}

	// Only support the Nanite vertex factory type.
	if (VertexFactoryData.VertexFactoryType != &FNaniteVertexFactory::StaticType)
	{
		return;
	}

	// Check if Nanite can be used by this material
	const FMaterialShadingModelField ShadingModels = Material.GetShadingModels();
	bool bShouldDraw = Nanite::IsSupportedBlendMode(Material) && Nanite::IsSupportedMaterialDomain(Material.GetMaterialDomain());
	if (!bShouldDraw)
	{
		return;
	}

	// Nanite passes always use the forced fixed vertex element and not custom default vertex declaration even if it's provided
	FPSOPrecacheVertexFactoryData NaniteVertexFactoryData = VertexFactoryData;
	NaniteVertexFactoryData.CustomDefaultVertexDeclaration = nullptr;

	CollectNanitePSOInitializers(SceneTexturesConfig, NaniteVertexFactoryData, Material, PreCacheParams, ShaderPlatform, PSOInitializers);
}

class FNaniteRasterPSOCollector : public FNaniteBasePSOCollector
{
public:

	FNaniteRasterPSOCollector(ERHIFeatureLevel::Type InFeatureLevel) : FNaniteBasePSOCollector(NaniteRasterPSOCollectorName, InFeatureLevel)
	{
	}

private:
	virtual void CollectNanitePSOInitializers(
		const FSceneTexturesConfig& SceneTexturesConfig,
		const FPSOPrecacheVertexFactoryData& VertexFactoryData,
		const FMaterial& Material,
		const FPSOPrecacheParams& PreCacheParams,
		EShaderPlatform ShaderPlatform,
		TArray<FPSOPrecacheData>& PSOInitializers) override
	{
		Nanite::CollectRasterPSOInitializers(SceneTexturesConfig, Material, PreCacheParams, ShaderPlatform, PSOCollectorIndex, PSOInitializers);
	}
};

class FNaniteShadingPSOCollector : public FNaniteBasePSOCollector
{
public:

	FNaniteShadingPSOCollector(ERHIFeatureLevel::Type InFeatureLevel) : FNaniteBasePSOCollector(NaniteShadingPSOCollectorName, InFeatureLevel)
	{
	}

private:
	virtual void CollectNanitePSOInitializers(
		const FSceneTexturesConfig& SceneTexturesConfig,
		const FPSOPrecacheVertexFactoryData& VertexFactoryData,
		const FMaterial& Material,
		const FPSOPrecacheParams& PreCacheParams,
		EShaderPlatform ShaderPlatform,
		TArray<FPSOPrecacheData>& PSOInitializers) override
	{
		Nanite::CollectBasePassShadingPSOInitializers(SceneTexturesConfig, VertexFactoryData, Material, PreCacheParams, FeatureLevel, ShaderPlatform, PSOCollectorIndex, PSOInitializers);
	}
};

class FNaniteLumenCardPSOCollector : public FNaniteBasePSOCollector
{
public:

	FNaniteLumenCardPSOCollector(ERHIFeatureLevel::Type InFeatureLevel) : FNaniteBasePSOCollector(NaniteLumenCardPSOCollectorName, InFeatureLevel)
	{
	}

private:
	virtual void CollectNanitePSOInitializers(
		const FSceneTexturesConfig& SceneTexturesConfig,
		const FPSOPrecacheVertexFactoryData& VertexFactoryData,
		const FMaterial& Material,
		const FPSOPrecacheParams& PreCacheParams,
		EShaderPlatform ShaderPlatform,
		TArray<FPSOPrecacheData>& PSOInitializers) override
	{
		Nanite::CollectLumenCardPSOInitializers(SceneTexturesConfig, VertexFactoryData, Material, PreCacheParams, FeatureLevel, ShaderPlatform, PSOCollectorIndex, PSOInitializers);
	}
};

IPSOCollector* CreateNaniteRasterPSOCollector(ERHIFeatureLevel::Type FeatureLevel)
{
	if (DoesPlatformSupportNanite(GetFeatureLevelShaderPlatform(FeatureLevel)))
	{
		return new FNaniteRasterPSOCollector(FeatureLevel);
	}
	else
	{
		return nullptr;
	}
}
FRegisterPSOCollectorCreateFunction RegisterNaniteRasterPSOCollector(&CreateNaniteRasterPSOCollector, EShadingPath::Deferred, NaniteRasterPSOCollectorName);

IPSOCollector* CreateNaniteShadingPSOCollector(ERHIFeatureLevel::Type FeatureLevel)
{
	if (DoesPlatformSupportNanite(GetFeatureLevelShaderPlatform(FeatureLevel)))
	{
		return new FNaniteShadingPSOCollector(FeatureLevel);
	}
	else
	{
		return nullptr;
	}
}
FRegisterPSOCollectorCreateFunction RegisterNaniteShadererPSOCollector(&CreateNaniteShadingPSOCollector, EShadingPath::Deferred, NaniteShadingPSOCollectorName);

IPSOCollector* CreateNaniteLumenCardPSOCollector(ERHIFeatureLevel::Type FeatureLevel)
{
	if (DoesPlatformSupportNanite(GetFeatureLevelShaderPlatform(FeatureLevel)))
	{
		return new FNaniteLumenCardPSOCollector(FeatureLevel);
	}
	else
	{
		return nullptr;
	}
}
FRegisterPSOCollectorCreateFunction RegisterNaniteLumenCardPSOCollector(&CreateNaniteLumenCardPSOCollector, EShadingPath::Deferred, NaniteLumenCardPSOCollectorName);

FNaniteMaterialSlot& FNaniteMaterialListContext::GetMaterialSlotForWrite(FPrimitiveSceneInfo& PrimitiveSceneInfo, ENaniteMeshPass::Type MeshPass, uint8 SectionIndex)
{	
	TArray<FNaniteMaterialSlot>& MaterialSlots = PrimitiveSceneInfo.NaniteMaterialSlots[MeshPass];

	// Initialize material slots if they haven't been already
	// NOTE: Lazily initializing them like this prevents adding material slots for primitives that have no bins in the pass
	if (MaterialSlots.Num() == 0)
	{
		check(PrimitiveSceneInfo.Proxy->IsNaniteMesh());
		check(PrimitiveSceneInfo.NaniteRasterBins[MeshPass].Num() == 0);
		check(PrimitiveSceneInfo.NaniteShadingBins[MeshPass].Num() == 0);

		auto* NaniteSceneProxy = static_cast<const Nanite::FSceneProxyBase*>(PrimitiveSceneInfo.Proxy);
		const int32 NumMaterialSections = NaniteSceneProxy->GetMaterialSections().Num();

		MaterialSlots.SetNumUninitialized(NumMaterialSections);
		FMemory::Memset(MaterialSlots.GetData(), 0xFF, NumMaterialSections * MaterialSlots.GetTypeSize());
	}

	check(MaterialSlots.IsValidIndex(SectionIndex));
	return MaterialSlots[SectionIndex];
}

void FNaniteMaterialListContext::AddShadingBin(
	FPrimitiveSceneInfo& PrimitiveSceneInfo,
	const FNaniteShadingBin& TriangleShadingBin,
	const FNaniteShadingBin& VoxelShadingBin,
	const FNaniteShadingBin& CurveShadingBin,
	ENaniteMeshPass::Type MeshPass,
	uint8 SectionIndex)
{
	FNaniteMaterialSlot& MaterialSlot = GetMaterialSlotForWrite(PrimitiveSceneInfo, MeshPass, SectionIndex);
	check(MaterialSlot.TriangleShadingBin == 0xFFFFu);
	check(MaterialSlot.VoxelShadingBin == 0xFFFFu);
	check(MaterialSlot.CurveShadingBin == 0xFFFFu);

	MaterialSlot.TriangleShadingBin = TriangleShadingBin.BinIndex;
	MaterialSlot.VoxelShadingBin = VoxelShadingBin.BinIndex;
	MaterialSlot.CurveShadingBin = CurveShadingBin.BinIndex;

	PrimitiveSceneInfo.NaniteShadingBins[MeshPass].Add(TriangleShadingBin);

	if (VoxelShadingBin.IsValid())
	{
		PrimitiveSceneInfo.NaniteShadingBins[MeshPass].Add(VoxelShadingBin);
	}
	if (CurveShadingBin.IsValid())
	{
		PrimitiveSceneInfo.NaniteShadingBins[MeshPass].Add(CurveShadingBin);
	}
}

void FNaniteMaterialListContext::AddRasterBin(
	FPrimitiveSceneInfo& PrimitiveSceneInfo,
	const FNaniteRasterBin& PrimaryRasterBin,
	const FNaniteRasterBin& FallbackRasterBin,
	ENaniteMeshPass::Type MeshPass,
	uint8 SectionIndex)
{
	check(PrimaryRasterBin.IsValid());
	
	FNaniteMaterialSlot& MaterialSlot = GetMaterialSlotForWrite(PrimitiveSceneInfo, MeshPass, SectionIndex);
	check(MaterialSlot.RasterBin == 0xFFFFu);
	MaterialSlot.RasterBin = PrimaryRasterBin.BinIndex;
	MaterialSlot.FallbackRasterBin = FallbackRasterBin.BinIndex;
	
	PrimitiveSceneInfo.NaniteRasterBins[MeshPass].Add(PrimaryRasterBin);
	if (FallbackRasterBin.IsValid())
	{
		PrimitiveSceneInfo.NaniteRasterBins[MeshPass].Add(FallbackRasterBin);
	}
}

void FNaniteMaterialListContext::Apply(FScene& Scene)
{
	check(IsInParallelRenderingThread());

	struct FPSOCompilationData
	{
		FGraphEventArray AsyncCompileEvents;
		bool bRequestStaticMeshUpdate = false;
		bool bAddToLumenScene = false;
	};

	// Collect all the primitives which still have async PSO compilation for optional nanite rendering PSOs and
	// add task to schedule a static mesh update when the PSO compilation is done.
	TMap<FPrimitiveSceneInfo*, FPSOCompilationData> AsyncPSOCompileEventsPerPrimitiveSceneIndex;

	Nanite::FMaterialsSceneExtension* MaterialsExtension = Scene.GetExtensionPtr<Nanite::FMaterialsSceneExtension>();

	for (int32 MeshPassIndex = 0; MeshPassIndex < ENaniteMeshPass::Num; ++MeshPassIndex)
	{
		FNaniteRasterPipelines& RasterPipelines = Scene.NaniteRasterPipelines[MeshPassIndex];
		FNaniteShadingPipelines& ShadingPipelines = Scene.NaniteShadingPipelines[MeshPassIndex];
		FNaniteVisibility& Visibility = Scene.NaniteVisibility[MeshPassIndex];

		for (const FDeferredPipelines& PipelinesCommand : DeferredPipelines[MeshPassIndex])
		{
			FPrimitiveSceneInfo* PrimitiveSceneInfo = PipelinesCommand.PrimitiveSceneInfo;
			FNaniteVisibility::PrimitiveRasterBinType* RasterBins = Visibility.GetRasterBinReferences(PrimitiveSceneInfo);
			FNaniteVisibility::PrimitiveShadingBinType* ShadingBins = Visibility.GetShadingBinReferences(PrimitiveSceneInfo);
			
			// Clear the previous request if set (can be marked pending again if still compiling)
			if (MaterialsExtension)
			{
				MaterialsExtension->ClearPendingRequestStaticMeshUpdate(PrimitiveSceneInfo);
			}

			const int32 MaterialSectionCount = PipelinesCommand.MaterialSectionData.Num();
			for (int32 MaterialSectionIndex = 0; MaterialSectionIndex < MaterialSectionCount; ++MaterialSectionIndex)
			{
				const FMaterialSectionData& MaterialSectionData = PipelinesCommand.MaterialSectionData[MaterialSectionIndex];
				
				// Check if the material section has valid shading pipeline, otherwise don't bother to add the shading and raster bins
				// (but still collect the graph events for async update when PSOs are done compiling)
				bool bHasShadingPipeline = MaterialSectionData.TriangleShadingAsyncPSOCompilationEvents.IsEmpty();

				const FNaniteRasterPipeline& RasterPipeline = MaterialSectionData.RasterPipeline;

				// Register raster bin
				{

					FNaniteRasterBin PrimaryRasterBin;
					FNaniteRasterBin FallbackRasterBin;

					// Make sure PSOs are not compiling anymore	- if so fallback to fixed function
					if (MaterialsExtension && !MaterialSectionData.RasterAsyncPSOCompilationEvents.IsEmpty())
					{
						FNaniteRasterPipeline FixedFunctionRasterPipeline = FNaniteRasterPipeline::GetFixedFunctionPipeline(RasterPipeline.GetFixedFunctionBinMask());

						// Accumulate all the async graph events for primitive scene info so we wait for all them to be finished
						FPSOCompilationData& PSOCompilationData = AsyncPSOCompileEventsPerPrimitiveSceneIndex.FindOrAdd(PrimitiveSceneInfo);
						PSOCompilationData.bRequestStaticMeshUpdate = true;
						PSOCompilationData.AsyncCompileEvents.Append(MaterialSectionData.RasterAsyncPSOCompilationEvents);

						if (bHasShadingPipeline)
						{
							// Primary raster bin becomes the fixed function raster pipeline
							PrimaryRasterBin = RasterPipelines.Register(FixedFunctionRasterPipeline);
						}
					}
					else if (bHasShadingPipeline)
					{
						PrimaryRasterBin = RasterPipelines.Register(RasterPipeline);

						// Check to register a fallback bin (used to disable programmable functionality at a distance)						
						FNaniteRasterPipeline FallbackRasterPipeline;
						if (GNaniteAllowProgrammableDistances && RasterPipeline.GetFallbackPipeline(FallbackRasterPipeline))
						{
							FallbackRasterBin = RasterPipelines.Register(FallbackRasterPipeline);
						}
					}

					if (bHasShadingPipeline)
					{
						AddRasterBin(*PrimitiveSceneInfo, PrimaryRasterBin, FallbackRasterBin, ENaniteMeshPass::Type(MeshPassIndex), uint8(MaterialSectionIndex));

						if (RasterBins)
						{
							RasterBins->Add(FNaniteVisibility::FRasterBin { PrimaryRasterBin.BinIndex, FallbackRasterBin.BinIndex });
						}
					}
				}

				// Register shading bin
				if (!RasterPipeline.bTranslucent)
				{
					if (bHasShadingPipeline || MaterialsExtension == nullptr)
					{
						const FNaniteShadingPipeline& TriangleShadingPipeline = MaterialSectionData.TriangleShadingPipeline;
						const FNaniteShadingBin TriangleShadingBin = ShadingPipelines.Register(TriangleShadingPipeline);

						FNaniteShadingBin VoxelShadingBin;
						const FNaniteShadingPipeline& VoxelShadingPipeline = MaterialSectionData.VoxelShadingPipeline;
						if (VoxelShadingPipeline.ComputeShader)
						{
							VoxelShadingBin = ShadingPipelines.Register(VoxelShadingPipeline);
						}
					
						FNaniteShadingBin CurveShadingBin;
						const FNaniteShadingPipeline& CurveShadingPipeline = MaterialSectionData.CurveShadingPipeline;
						if (CurveShadingPipeline.ComputeShader)
						{
							CurveShadingBin = ShadingPipelines.Register(CurveShadingPipeline);
						}

						AddShadingBin(*PrimitiveSceneInfo, TriangleShadingBin, VoxelShadingBin, CurveShadingBin, ENaniteMeshPass::Type(MeshPassIndex), uint8(MaterialSectionIndex));

						if (ShadingBins)
						{
							ShadingBins->Add(FNaniteVisibility::FShadingBin { TriangleShadingBin.BinIndex, VoxelShadingBin.BinIndex, CurveShadingBin.BinIndex });
						}
					}
					else
					{
						check(MeshPassIndex == ENaniteMeshPass::LumenCardCapture);

						FPSOCompilationData& PSOCompilationData = AsyncPSOCompileEventsPerPrimitiveSceneIndex.FindOrAdd(PrimitiveSceneInfo);
						PSOCompilationData.bAddToLumenScene = true;
						PSOCompilationData.bRequestStaticMeshUpdate = true;
						PSOCompilationData.AsyncCompileEvents.Append(MaterialSectionData.TriangleShadingAsyncPSOCompilationEvents);
					}
				}
			}

			// This will register the primitive's raster bins for custom depth, if necessary
			if (MeshPassIndex == ENaniteMeshPass::BasePass)
			{
				PrimitiveSceneInfo->RefreshNaniteRasterBins();
			}
		}
	}

	for (auto Iter : AsyncPSOCompileEventsPerPrimitiveSceneIndex)
	{
		FPrimitiveSceneInfo* PrimitiveSceneInfo = Iter.Key;
		FPSOCompilationData& PSOCompilationData = Iter.Value;

		// Nanite Material extention should be valid when there async PSO outstanding
		check(MaterialsExtension);
		TSharedPtr<bool> RequestValid = MaterialsExtension->AddPendingRequestStaticMeshUpdate(PrimitiveSceneInfo);

		FFunctionGraphTask::CreateAndDispatchWhenReady(
			[&Scene, RequestValid, bRequestStaticMeshUpdate = PSOCompilationData.bRequestStaticMeshUpdate, bAddToLumenScene = PSOCompilationData.bAddToLumenScene, PersistentPrimitiveIndex = PrimitiveSceneInfo->GetPersistentIndex()]()
			{
				// No race on reading this bool because it all happens on render thread
				bool bRequestValid = *RequestValid.Get();
				if (bRequestValid)
				{
					FPrimitiveSceneInfo* PrimitiveSceneInfo = Scene.GetPrimitiveSceneInfo(PersistentPrimitiveIndex);
					if (PrimitiveSceneInfo)
					{
						if (bRequestStaticMeshUpdate)
						{
							PrimitiveSceneInfo->RequestStaticMeshUpdate();
						}
					}

					if (auto Extension = Scene.GetExtensionPtr<Nanite::FMaterialsSceneExtension>())
					{
						Extension->ClearPendingRequestStaticMeshUpdate(PrimitiveSceneInfo);
					}
				}
			}, TStatId(), &PSOCompilationData.AsyncCompileEvents, ENamedThreads::GetRenderThread_Local());
	}
}
