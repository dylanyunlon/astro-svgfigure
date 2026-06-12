// Copyright Epic Games, Inc. All Rights Reserved.

#include "CoreMinimal.h"
#include "ComponentRecreateRenderStateContext.h"
#include "Containers/AllowShrinking.h"
#include "Containers/ArrayView.h"
#include "Containers/BinaryHeap.h"
#include "Containers/SetUtilities.h"
#include "Containers/SparseSet.h"
#include "Containers/StaticArray.h"
#include "DistanceFieldLightingShared.h"
#include "Engine/EngineTypes.h"
#include "Engine/Scene.h"
#include "Experimental/ConcurrentLinearAllocator.h"
#include "Experimental/Containers/RobinHoodHashTable.h"
#include "GlobalDistanceField.h"
#include "GlobalRenderResources.h"
#include "HAL/IConsoleManager.h"
#include "HAL/LowLevelMemTracker.h"
#include "Hash/xxhash.h"
#include "InstanceDataSceneProxy.h"
#include "Logging/LogMacros.h"
#include "Lumen/Lumen.h"
#include "Lumen/LumenMeshCards.h"
#include "Lumen/LumenSceneData.h"
#include "Lumen/LumenSceneGPUDrivenUpdate.h"
#include "Lumen/LumenSparseSpanArray.h"
#include "Lumen/LumenSurfaceCacheFeedback.h"
#include "Lumen/LumenUniqueList.h"
#include "Lumen/LumenViewState.h"
#include "MeshCardBuild.h"
#include "Misc/LargeWorldCoordinates.h"
#include "Misc/LargeWorldRenderPosition.h"
#include "MultiGPU.h"
#include "Nanite/NaniteShared.h"
#include "NaniteSceneProxy.h"
#include "PrimitiveComponentId.h"
#include "PrimitiveSceneInfo.h"
#include "PrimitiveSceneProxy.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "ProfilingDebugging/TagTrace.h"
#include "RayTracing/RaytracingOptions.h"
#include "RendererInterface.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RenderResource.h"
#include "RenderTransform.h"
#include "RenderUtils.h"
#include "RHI.h"
#include "RHIBreadcrumbs.h"
#include "RHICommandList.h"
#include "RHIContext.h"
#include "RHIResources.h"
#include "RHIStaticStates.h"
#include "ScenePrivate.h"
#include "SceneRendering.h"
#include "SceneView.h"
#include "Shader.h"
#include "ShaderParameterMacros.h"
#include "ShowFlags.h"
#include "Stats/Stats.h"
#include "UnifiedBuffer.h"
#include "VirtualTexturing.h"
#include "VT/RuntimeVirtualTextureEnum.h"
#include "VT/RuntimeVirtualTextureSceneProxy.h"
#include "SceneViewState.h"
#include "LogRenderer.h"

int32 GLumenSceneGlobalDFResolution = 252;
FAutoConsoleVariableRef CVarLumenSceneGlobalDFResolution(
	TEXT("r.LumenScene.GlobalSDF.Resolution"),
	GLumenSceneGlobalDFResolution,
	TEXT("Global Distance Field resolution when Lumen is enabled."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenSceneGlobalDFClipmapExtent = 2500.0f;
FAutoConsoleVariableRef CVarLumenSceneGlobalDFClipmapExtent(
	TEXT("r.LumenScene.GlobalSDF.ClipmapExtent"),
	GLumenSceneGlobalDFClipmapExtent,
	TEXT("Global Distance Field first clipmap extent when Lumen is enabled."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenFarField(
	TEXT("r.LumenScene.FarField"),
	0,
	TEXT("Enable/Disable Lumen far-field ray tracing."),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		// Recreate proxies so that FPrimitiveSceneProxy::UpdateVisibleInLumenScene() can pick up any changed state
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarLumenFarFieldOcclusionOnly(
	TEXT("r.LumenScene.FarField.OcclusionOnly"),
	0,
	TEXT("Whether to speedup Far Field traces by only calculating occlusion."),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
		{
			// Recreate proxies so that FPrimitiveSceneProxy::UpdateVisibleInLumenScene() can pick up any changed state
			FGlobalComponentRecreateRenderStateContext Context;
		}),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<float> CVarLumenFarFieldMaxTraceDistance(
	TEXT("r.LumenScene.FarField.MaxTraceDistance"),
	1.0e6f,
	TEXT("Maximum hit-distance for Lumen far-field ray tracing (Default = 1.0e6)."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<float> CVarLumenFarFieldDitherScale(
	TEXT("r.LumenScene.FarField.FarFieldDitherScale"),
	200.0f,
	TEXT("Dither region between near and far field in world space units."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

TAutoConsoleVariable<int32> CVarLumenSceneUpdateViewOrigin(
	TEXT("r.LumenScene.UpdateViewOrigin"),
	1,
	TEXT("Whether to update view origin for voxel lighting and global distance field. Useful for debugging."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenSceneSurfaceCacheAtlasSize(
	TEXT("r.LumenScene.SurfaceCache.AtlasSize"),
	4096,
	TEXT("Surface cache card atlas size."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

namespace Lumen
{
	bool UseFarField(const FSceneViewFamily& ViewFamily)
	{
		return CVarLumenFarField.GetValueOnRenderThread() != 0 
			&& ViewFamily.EngineShowFlags.LumenFarFieldTraces;
	}

	bool UseFarFieldOcclusionOnly()
	{
		return CVarLumenFarFieldOcclusionOnly.GetValueOnAnyThread() != 0;
	}

	float GetFarFieldMaxTraceDistance()
	{
		return CVarLumenFarFieldMaxTraceDistance.GetValueOnRenderThread();
	}

	float GetNearFieldMaxTraceDistanceDitherScale(bool bUseFarField)
	{
		return bUseFarField ? CVarLumenFarFieldDitherScale.GetValueOnRenderThread() : 0.0f;
	}

	float GetNearFieldSceneRadius(const FViewInfo& View, bool bUseFarField)
	{
		float SceneRadius = FLT_MAX;

		if (bUseFarField && RayTracing::GetCullingMode(View.Family->EngineShowFlags) != RayTracing::ECullingMode::Disabled)
		{
			return GetRayTracingCullingRadius();
		}

		return SceneRadius;
	}

	bool SetLandscapeHeightSamplingParameters(const FVector& LumenSceneViewOrigin, const FScene* Scene, FLumenLandscapeHeightSamplingParameters& OutParameters)
	{
		bool bSetToDefault = true;
		const FRuntimeVirtualTextureSceneProxy* HeightVirtualTextureProxy = nullptr;

		if (Scene)
		{
			for (const FRuntimeVirtualTextureSceneProxy* VirtualTextureProxy : Scene->RuntimeVirtualTextures)
			{
				if (VirtualTextureProxy
					&& VirtualTextureProxy->GetMaterialType() == ERuntimeVirtualTextureMaterialType::WorldHeight
					&& VirtualTextureProxy->GetBounds().IsInsideXY(LumenSceneViewOrigin))
				{
					HeightVirtualTextureProxy = VirtualTextureProxy;
					break;
				}
			}
		}

		if (HeightVirtualTextureProxy)
		{
			if (const IAllocatedVirtualTexture* AllocatedTexture = HeightVirtualTextureProxy->GetAllocatedVirtualTexture())
			{
				const uint32 LayerIndex = 0;
				const uint32 PageTableIndex = 0;

				if (FRHIShaderResourceView* PhysicalTextureSRV = AllocatedTexture->GetPhysicalTextureSRV(LayerIndex, false))
				{
					OutParameters.HeightVirtualTexture = PhysicalTextureSRV;
					AllocatedTexture->GetPackedUniform(&OutParameters.HeightVirtualTextureUniforms, LayerIndex);
				}

				OutParameters.HeightVirtualTexturePageTable = PageTableIndex < AllocatedTexture->GetNumPageTableTextures() ?
					AllocatedTexture->GetPageTableTexture(PageTableIndex) : nullptr;
				OutParameters.HeightVirtualTexturePageTableIndirection = AllocatedTexture->GetPageTableIndirectionTexture();
				OutParameters.HeightVirtualTextureAdaptive = HeightVirtualTextureProxy->IsAdaptive();
				OutParameters.HeightVirtualTextureSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();

				if (OutParameters.HeightVirtualTexturePageTable)
				{
					FUintVector4 PageTableUniforms[2];
					AllocatedTexture->GetPackedPageTableUniform(PageTableUniforms);

					OutParameters.HeightVirtualTexturePackedUniform0 = PageTableUniforms[0];
					OutParameters.HeightVirtualTexturePackedUniform1 = PageTableUniforms[1];

					FVector4 WorldToUVParameters[4];
					WorldToUVParameters[0] = HeightVirtualTextureProxy->GetUniformParameter(ERuntimeVirtualTextureShaderUniform_WorldToUVTransform0);
					WorldToUVParameters[1] = HeightVirtualTextureProxy->GetUniformParameter(ERuntimeVirtualTextureShaderUniform_WorldToUVTransform1);
					WorldToUVParameters[2] = HeightVirtualTextureProxy->GetUniformParameter(ERuntimeVirtualTextureShaderUniform_WorldToUVTransform2);
					WorldToUVParameters[3] = HeightVirtualTextureProxy->GetUniformParameter(ERuntimeVirtualTextureShaderUniform_WorldHeightUnpack);

					FLargeWorldRenderPosition HeightVirtualTextureOrigin(WorldToUVParameters[0]);
					OutParameters.HeightVirtualTextureLWCTile = HeightVirtualTextureOrigin.GetTile();

					OutParameters.HeightVirtualTextureWorldToUVTransform = FMatrix44f(
						HeightVirtualTextureOrigin.GetOffset(),
						FVector3f((FVector4f)WorldToUVParameters[1]),
						FVector3f((FVector4f)WorldToUVParameters[2]),
						FVector3f((FVector4f)WorldToUVParameters[3]));

					OutParameters.HeightVirtualTextureEnabled = 1;
				}

				bSetToDefault = !OutParameters.HeightVirtualTexture
					|| !OutParameters.HeightVirtualTexturePageTable
					|| (OutParameters.HeightVirtualTextureAdaptive != 0 && !OutParameters.HeightVirtualTexturePageTableIndirection);
			}
		}

		if (bSetToDefault)
		{
			OutParameters.HeightVirtualTexture = GBlackTextureWithSRV->ShaderResourceViewRHI;
			OutParameters.HeightVirtualTexturePageTable = GBlackUintTexture->TextureRHI;
			OutParameters.HeightVirtualTexturePageTableIndirection = GBlackUintTexture->TextureRHI;
			OutParameters.HeightVirtualTextureAdaptive = 0;
			OutParameters.HeightVirtualTextureLWCTile = FVector3f::ZeroVector;
			OutParameters.HeightVirtualTextureWorldToUVTransform = FMatrix44f::Identity;
			OutParameters.HeightVirtualTextureEnabled = 0;
			OutParameters.HeightVirtualTexturePackedUniform0 = FUintVector4::ZeroValue;
			OutParameters.HeightVirtualTexturePackedUniform1 = FUintVector4::ZeroValue;
			OutParameters.HeightVirtualTextureUniforms = FUintVector4::ZeroValue;
		}

		return !bSetToDefault;
	}
}

int32 Lumen::GetGlobalDFResolution()
{
	return GLumenSceneGlobalDFResolution;
}

float Lumen::GetGlobalDFClipmapExtent(int32 ClipmapIndex)
{
	return GLumenSceneGlobalDFClipmapExtent * FMath::Pow(2.0f, ClipmapIndex);
}

int32 Lumen::GetNumGlobalDFClipmaps(const FSceneView& View)
{
	return GlobalDistanceField::GetNumGlobalDistanceFieldClipmaps(/*bLumenEnabled*/ true, View.FinalPostProcessSettings.LumenSceneViewDistance);
}

bool Lumen::ShouldUpdateLumenSceneViewOrigin()
{
	return CVarLumenSceneUpdateViewOrigin.GetValueOnRenderThread() != 0;
}

FVector Lumen::GetLumenSceneViewOrigin(const FViewInfo& View, int32 ClipmapIndex)
{
	FVector CameraOrigin = View.ViewMatrices.GetViewOrigin();

	if (View.ViewState)
	{
		FVector CameraVelocityOffset = View.ViewState->GlobalDistanceFieldData->CameraVelocityOffset;

		if (ClipmapIndex > 0)
		{
			const float ClipmapExtent = Lumen::GetGlobalDFClipmapExtent(ClipmapIndex);
			const float MaxCameraDriftFraction = .75f;
			CameraVelocityOffset.X = FMath::Clamp<float>(CameraVelocityOffset.X, -ClipmapExtent * MaxCameraDriftFraction, ClipmapExtent * MaxCameraDriftFraction);
			CameraVelocityOffset.Y = FMath::Clamp<float>(CameraVelocityOffset.Y, -ClipmapExtent * MaxCameraDriftFraction, ClipmapExtent * MaxCameraDriftFraction);
			CameraVelocityOffset.Z = FMath::Clamp<float>(CameraVelocityOffset.Z, -ClipmapExtent * MaxCameraDriftFraction, ClipmapExtent * MaxCameraDriftFraction);
		}

		CameraOrigin += CameraVelocityOffset;
	}

	// Frozen camera
	if (View.ViewState)
	{
		if (Lumen::ShouldUpdateLumenSceneViewOrigin())
		{
			View.ViewState->GlobalDistanceFieldData->bUpdateViewOrigin = true;
		}
		else
		{
			if (View.ViewState->GlobalDistanceFieldData->bUpdateViewOrigin)
			{
				View.ViewState->GlobalDistanceFieldData->LastViewOrigin = View.ViewMatrices.GetViewOrigin();
				View.ViewState->GlobalDistanceFieldData->bUpdateViewOrigin = false;
			}
		}

		if (!View.ViewState->GlobalDistanceFieldData->bUpdateViewOrigin)
		{
			CameraOrigin = View.ViewState->GlobalDistanceFieldData->LastViewOrigin;
		}
	}

	return CameraOrigin;
}

class FLumenCardPageGPUData
{
public:
	// Must match usf
	enum { DataStrideInFloat4s = 5 };
	enum { DataStrideInBytes = DataStrideInFloat4s * sizeof(FVector4f) };

	static void FillData(const FLumenPageTableEntry& RESTRICT PageTableEntry, uint32 ResLevelPageTableOffset, FIntPoint ResLevelSizeInTiles, FVector2D InvPhysicalAtlasSize, FVector4f* RESTRICT OutData)
	{
		// Layout must match GetLumenCardPageData in usf
		const float SizeInTexelsX = PageTableEntry.PhysicalAtlasRect.Max.X - PageTableEntry.PhysicalAtlasRect.Min.X;
		const float SizeInTexelsY = PageTableEntry.PhysicalAtlasRect.Max.Y - PageTableEntry.PhysicalAtlasRect.Min.Y;

		OutData[0].X = *(float*)&PageTableEntry.CardIndex;
		OutData[0].Y = *(float*)&ResLevelPageTableOffset;
		OutData[0].Z = SizeInTexelsX;
		OutData[0].W = SizeInTexelsY;

		OutData[1] = PageTableEntry.CardUVRect;

		OutData[2].X = PageTableEntry.PhysicalAtlasRect.Min.X * InvPhysicalAtlasSize.X;
		OutData[2].Y = PageTableEntry.PhysicalAtlasRect.Min.Y * InvPhysicalAtlasSize.Y;
		OutData[2].Z = PageTableEntry.PhysicalAtlasRect.Max.X * InvPhysicalAtlasSize.X;
		OutData[2].W = PageTableEntry.PhysicalAtlasRect.Max.Y * InvPhysicalAtlasSize.Y;

		OutData[3].X = SizeInTexelsX > 0.0f ? ((PageTableEntry.CardUVRect.Z - PageTableEntry.CardUVRect.X) / SizeInTexelsX) : 0.0f;
		OutData[3].Y = SizeInTexelsY > 0.0f ? ((PageTableEntry.CardUVRect.W - PageTableEntry.CardUVRect.Y) / SizeInTexelsY) : 0.0f;
		OutData[3].Z = *(float*)&ResLevelSizeInTiles.X;
		OutData[3].W = *(float*)&ResLevelSizeInTiles.Y;

		const uint32 LastUpdateFrame = 0;
		OutData[4].X = *(float*)&LastUpdateFrame;
		OutData[4].Y = *(float*)&LastUpdateFrame;
		OutData[4].Z = *(float*)&LastUpdateFrame;
		// This is only used to rotate through the texels in a quad when using adaptive direct lighting shadow rays.
		// So we can store a TemporalIndexMod4 and use only 2 bits if needed.
		OutData[4].W = *(float*)&LastUpdateFrame;

		static_assert(DataStrideInFloat4s == 5, "Data stride doesn't match");
	}
};

static FIntPoint GetDesiredPhysicalAtlasSizeInPages(float SurfaceCacheResolution)
{
	int32 AtlasSizeInPages = FMath::DivideAndRoundUp<uint32>(CVarLumenSceneSurfaceCacheAtlasSize.GetValueOnRenderThread(), Lumen::PhysicalPageSize);
	AtlasSizeInPages = AtlasSizeInPages * SurfaceCacheResolution;
	AtlasSizeInPages = FMath::Clamp(AtlasSizeInPages, 1, 64);
	return FIntPoint(AtlasSizeInPages, AtlasSizeInPages);
}

static FIntPoint GetDesiredPhysicalAtlasSize(float SurfaceCacheResolution)
{
	return GetDesiredPhysicalAtlasSizeInPages(SurfaceCacheResolution) * Lumen::PhysicalPageSize;
}

FLumenPrimitiveGroupRemoveInfo::FLumenPrimitiveGroupRemoveInfo(const FPrimitiveSceneInfo* InPrimitive, int32 InPrimitiveIndex)
	: Primitive(InPrimitive)
	, PrimitiveIndex(InPrimitiveIndex)
	, LumenPrimitiveGroupIndices(InPrimitive->LumenPrimitiveGroupIndices)
{
}

FLumenPrimitiveGroup::~FLumenPrimitiveGroup()
{
	if (HasPrimitivesArray())
	{
		FreePrimitivesArray();
		PrimitivesArray = nullptr;
	}
}

FLumenPrimitiveGroup::FLumenPrimitiveGroup(const FLumenPrimitiveGroup& Other)
{
	*this = Other;
}

FLumenPrimitiveGroup::FLumenPrimitiveGroup(FLumenPrimitiveGroup&& Other)
{
	*this = MoveTemp(Other);
}

FLumenPrimitiveGroup& FLumenPrimitiveGroup::operator=(const FLumenPrimitiveGroup& Other)
{
	if (&Other != this)
	{
		if (HasPrimitivesArray())
		{
			FreePrimitivesArray();
		}

		FMemory::Memcpy(this, &Other, sizeof(FLumenPrimitiveGroup));

		if (HasPrimitivesArray())
		{
			PrimitivesArrayMax = DefaultCalculateSlackReserve(NumPrimitives, PrimitivesArrayElementSize, true);
			PrimitivesArray = (FPrimitiveSceneInfo**)FMemory::Malloc(PrimitivesArrayMax * PrimitivesArrayElementSize);
			FMemory::Memcpy(PrimitivesArray, Other.PrimitivesArray, NumPrimitives * PrimitivesArrayElementSize);
		}
	}

	return *this;
}

FLumenPrimitiveGroup& FLumenPrimitiveGroup::operator=(FLumenPrimitiveGroup&& Other)
{
	if (&Other != this)
	{
		if (HasPrimitivesArray())
		{
			FreePrimitivesArray();
		}

		FMemory::Memcpy(this, &Other, sizeof(FLumenPrimitiveGroup));

		new (&Other) FLumenPrimitiveGroup;
	}

	return *this;
}

void FLumenPrimitiveGroup::AddPrimitive(FPrimitiveSceneInfo* PrimitiveSceneInfo)
{
	if (NumPrimitives == 0u)
	{
		check(PrimitiveInlined == nullptr);
		PrimitiveInlined = PrimitiveSceneInfo;
	}
	else
	{
		if (NumPrimitives + 1u > PrimitivesArrayMax)
		{
			const bool bInlined = !HasPrimitivesArray();

			PrimitivesArrayMax = DefaultCalculateSlackGrow(NumPrimitives + 1u, PrimitivesArrayMax, PrimitivesArrayElementSize, true);
			auto NewPrimitivesArray = (FPrimitiveSceneInfo**)FMemory::Realloc(bInlined ? nullptr : PrimitivesArray, PrimitivesArrayMax * PrimitivesArrayElementSize);

			if (bInlined)
			{
				NewPrimitivesArray[0] = PrimitiveInlined;
			}
			PrimitivesArray = NewPrimitivesArray;
		}

		PrimitivesArray[NumPrimitives] = PrimitiveSceneInfo;
	}

	++NumPrimitives;
}

void FLumenPrimitiveGroup::RemovePrimitiveAtSwap(int32 Index, EAllowShrinking AllowShrinking)
{
	check(Index >= 0 && Index < int32(NumPrimitives));

	if (HasPrimitivesArray())
	{
		Swap(PrimitivesArray[Index], PrimitivesArray[NumPrimitives - 1u]);

		if (NumPrimitives - 1u == 1u)
		{
			FPrimitiveSceneInfo* PrimitiveToInline = PrimitivesArray[0];
			FreePrimitivesArray();
			PrimitiveInlined = PrimitiveToInline;
			PrimitivesArrayMax = 0u;
		}
		else if (AllowShrinking == EAllowShrinking::Yes)
		{
			const uint32 NewPrimitivesArrayMax = DefaultCalculateSlackShrink(NumPrimitives - 1u, PrimitivesArrayMax, PrimitivesArrayElementSize, true);
			if (NewPrimitivesArrayMax != PrimitivesArrayMax)
			{
				PrimitivesArrayMax = NewPrimitivesArrayMax;
				PrimitivesArray = (FPrimitiveSceneInfo**)FMemory::Realloc(PrimitivesArray, NewPrimitivesArrayMax * PrimitivesArrayElementSize);
			}
		}
	}
	else
	{
		PrimitiveInlined = nullptr;
	}

	--NumPrimitives;
}

TConstArrayView<FPrimitiveSceneInfo*> FLumenPrimitiveGroup::GetPrimitives() const
{
	return MakeConstArrayView(HasPrimitivesArray() ? PrimitivesArray : &PrimitiveInlined, NumPrimitives);
}

bool FLumenPrimitiveGroup::HasMergedInstances() const
{
	bool HasInstancesToMerge = false;

	if (PrimitiveInstanceIndex < 0)
	{
		// Check if there is more than 1 instance for merging

		uint32 NumInstances = 0;
		for (const FPrimitiveSceneInfo* PrimitiveSceneInfo : GetPrimitives())
		{
			NumInstances += PrimitiveSceneInfo->GetNumInstanceSceneDataEntries();

			if (NumInstances > 1)
			{
				HasInstancesToMerge = true;
				break;
			}
		}
	}

	return HasInstancesToMerge;
}

void FLumenPrimitiveGroup::FreePrimitivesArray()
{
	check(HasPrimitivesArray() && PrimitivesArrayMax >= NumPrimitives && PrimitivesArray != nullptr);
	FMemory::Free(PrimitivesArray);
}

FLumenSurfaceCacheAllocator::FPageBin::FPageBin(const FIntPoint& InElementSize)
{
	ensure(InElementSize.GetMax() <= Lumen::PhysicalPageSize);
	ElementSize = InElementSize;
	PageSizeInElements = FIntPoint(Lumen::PhysicalPageSize) / InElementSize;
}

void FLumenSurfaceCacheAllocator::Init(const FIntPoint& InPageAtlasSizeInPages)
{
	PageAtlasSizeInPages = InPageAtlasSizeInPages;
	PhysicalPageFreeCount = InPageAtlasSizeInPages.X * InPageAtlasSizeInPages.Y;
	PhysicalPageList.Init(false, PhysicalPageFreeCount);
	
	// Init() can be calls several time if Lumen is enabled/disabled several time during a session
	// PageBinLookup needs to be initialized only once. After that, its data needs to be kept as the PageBins 
	// are never deallocated, and PageBinLookup holds the lookup towards these PageBins.
	if (bInitPageBinLookup)
	{
		PageBinLookup = FPageBinLookup(InPlace, 0xFF /*InvalidPageBinIndex*/);
		bInitPageBinLookup = false;
	}
}

FIntPoint FLumenSurfaceCacheAllocator::AllocatePhysicalAtlasPage()
{
	const int32 LinearIndex = PhysicalPageList.FindAndSetFirstZeroBit();
	--PhysicalPageFreeCount;
	check(LinearIndex != INDEX_NONE);
	return FIntPoint(LinearIndex % PageAtlasSizeInPages.X, LinearIndex / PageAtlasSizeInPages.X);
}

void FLumenSurfaceCacheAllocator::FreePhysicalAtlasPage(const FIntPoint& PageCoord)
{
	const uint32 LinearIndex = PageCoord.X + PageCoord.Y * PageAtlasSizeInPages.X;
	++PhysicalPageFreeCount;
	check(PhysicalPageList.IsValidIndex(LinearIndex));
	PhysicalPageList[LinearIndex] = false;
}

void FLumenSurfaceCacheAllocator::Allocate(const FLumenPageTableEntry& Page, FAllocation& Allocation)
{
	if (Page.IsSubAllocation())
	{
		FPageBin* MatchingBin = GetOrAddBin(Page.SubAllocationSize);
		check(MatchingBin);

		FPageBinAllocation* MatchingBinAllocation = MatchingBin->GetBinAllocation();

		if (!MatchingBinAllocation)
		{
			MatchingBinAllocation = MatchingBin->AddBinAllocation(AllocatePhysicalAtlasPage());
		}

		if (MatchingBinAllocation)
		{
			const FIntPoint ElementCoord = MatchingBinAllocation->Add();
			check(MatchingBinAllocation->PageCoord.X >= 0 && MatchingBinAllocation->PageCoord.Y >= 0);

			const FIntPoint ElementOffset = MatchingBinAllocation->PageCoord * Lumen::PhysicalPageSize + ElementCoord * MatchingBin->ElementSize;

			Allocation.PhysicalPageCoord = MatchingBinAllocation->PageCoord;
			Allocation.PhysicalAtlasRect.Min = ElementOffset;
			Allocation.PhysicalAtlasRect.Max = ElementOffset + MatchingBin->ElementSize;
		}
	}
	else
	{
		Allocation.PhysicalPageCoord = AllocatePhysicalAtlasPage();
		Allocation.PhysicalAtlasRect.Min = (Allocation.PhysicalPageCoord + 0) * Lumen::PhysicalPageSize;
		Allocation.PhysicalAtlasRect.Max = (Allocation.PhysicalPageCoord + 1) * Lumen::PhysicalPageSize;
	}
}

void FLumenSurfaceCacheAllocator::Free(const FLumenPageTableEntry& Page)
{
	if (Page.IsSubAllocation())
	{
		FPageBin* MatchingBin = GetBin(Page.SubAllocationSize);
		check(MatchingBin);
		checkSlow(Page.SubAllocationSize == MatchingBin->ElementSize);

		if (MatchingBin->RemoveBinAllocation(Page))
		{
			FreePhysicalAtlasPage(Page.PhysicalPageCoord);
		}
	}
	else
	{
		FreePhysicalAtlasPage(Page.PhysicalPageCoord);
	}
}

/**
 * Checks if there's enough free memory in the surface cache to allocate entire mip map level of a card (or a single page)
 */
bool FLumenSurfaceCacheAllocator::IsSpaceAvailable(const FLumenCard& Card, int32 ResLevel, bool bSinglePage) const
{
	FLumenMipMapDesc MipMapDesc;

	Card.GetMipMapDesc(ResLevel, MipMapDesc);

	const int32 ReqSizeInPages = bSinglePage ? 1 : (MipMapDesc.SizeInPages.X * MipMapDesc.SizeInPages.Y);

	if (PhysicalPageFreeCount >= ReqSizeInPages)
	{
		return true;
	}

	// No free pages, but maybe there's some space in one of the existing bins
	if (MipMapDesc.bSubAllocation)
	{
		if (const FPageBin* MatchingBin = GetBin(MipMapDesc.Resolution))
		{
			return MatchingBin->HasFreeElements();
		}
	}

	return false;
}

void FLumenSurfaceCacheAllocator::GetStats(FStats& Stats) const
{
	Stats.NumFreePages = PhysicalPageFreeCount;

	for (const FPageBin& Bin : PageBins)
	{
		const uint32 NumFreeElements = Bin.GetSubPageFreeCount();
		const uint32 NumElementsPerPage = Bin.PageSizeInElements.X * Bin.PageSizeInElements.Y;
		const uint32 NumElements = Bin.GetBinAllocationCount() * NumElementsPerPage - NumFreeElements;

		Stats.BinNumPages += Bin.GetBinAllocationCount();
		Stats.BinNumWastedPages += Bin.GetBinAllocationCount() - FMath::DivideAndRoundUp(NumElements, NumElementsPerPage);
		Stats.BinPageFreeTexels += NumFreeElements * Bin.ElementSize.X * Bin.ElementSize.Y;

		if (NumElements > 0)
		{
			FBinStats& NewBinStats = Stats.Bins.AddDefaulted_GetRef();
			NewBinStats.ElementSize = Bin.ElementSize;
			NewBinStats.NumAllocations = NumElements;
			NewBinStats.NumPages = Bin.GetBinAllocationCount();
		}
	}

	struct FSortBySize
	{
		FORCEINLINE bool operator()(const FBinStats& A, const FBinStats& B) const
		{
			const int32 AreaA = A.ElementSize.X * A.ElementSize.Y;
			const int32 AreaB = B.ElementSize.X * B.ElementSize.Y;

			if (AreaA == AreaB)
			{
				if (A.ElementSize.X == B.ElementSize.X)
				{
					return A.ElementSize.Y < B.ElementSize.Y;
				}
				else
				{
					return A.ElementSize.X < B.ElementSize.X;
				}
			}

			return AreaA < AreaB;
		}
	};

	Stats.Bins.Sort(FSortBySize());
}

void FLumenSceneData::UploadPageTable(FRDGBuilder& GraphBuilder, FRDGScatterUploadBuilder& UploadBuilder, FLumenSceneFrameTemporaries& FrameTemporaries)
{
	RDG_EVENT_SCOPE(GraphBuilder, "LumenUploadPageTable");
	RDG_GPU_MASK_SCOPE(GraphBuilder, FRHIGPUMask::All());

	if (bReuploadSceneRequest)
	{
		PageTableIndicesToUpdateInBuffer.SetNum(PageTable.Num());

		for (int32 PageIndex = 0; PageIndex < PageTable.Num(); ++PageIndex)
		{
			PageTableIndicesToUpdateInBuffer[PageIndex] = PageIndex;
		}
	}

	const uint32 NumElements = FMath::Max(1024u, FMath::RoundUpToPowerOfTwo(PageTable.Num()));
	const int32 NumElementsToUpload = PageTableIndicesToUpdateInBuffer.Num();

	// PageTableBuffer
	{
		const int32 NumBytesPerElement = 2 * sizeof(uint32);
		FRDGBuffer* PageTableBufferRDG = ResizeByteAddressBufferIfNeeded(GraphBuilder, PageTableBuffer, NumElements * NumBytesPerElement, TEXT("Lumen.PageTable"));
		FrameTemporaries.PageTableBufferSRV = GraphBuilder.CreateSRV(PageTableBufferRDG);

		if (NumElementsToUpload > 0)
		{
			UploadBuilder.AddPass(
				GraphBuilder,
				PageTableUploadBuffer,
				PageTableBufferRDG,
				NumElementsToUpload,
				NumBytesPerElement,
				TEXT("Lumen.PageTableUpload"),
				[this] (FRDGScatterUploader& Uploader)
			{
				for (int32 PageIndex : PageTableIndicesToUpdateInBuffer)
				{
					if (PageIndex < PageTable.Num())
					{
						uint32 PackedData[2] = { 0, 0 };

						if (PageTable.IsAllocated(PageIndex))
						{
							const FLumenPageTableEntry& Page = PageTable[PageIndex];

							PackedData[0] |= ((Page.SampleAtlasBiasX & 0xFFF) << 0);
							PackedData[0] |= ((Page.SampleAtlasBiasY & 0xFFF) << 12);
							PackedData[0] |= ((Page.SampleCardResLevelX & 0xF) << 24);
							PackedData[0] |= ((Page.SampleCardResLevelY & 0xF) << 28);

							PackedData[1] = Page.SamplePageIndex;
						}

						Uploader.Add(PageIndex, PackedData);
					}
				}
			});
		}
	}

	// CardPageBuffer
	{
		TRefCountPtr<FRDGPooledBuffer> CardPageBufferOld = CardPageBuffer;

		const int32 NumBytesPerElement = FLumenCardPageGPUData::DataStrideInFloat4s * sizeof(FVector4f);
		FRDGBuffer* CardPageBufferRDG = ResizeStructuredBufferIfNeeded(GraphBuilder, CardPageBuffer, NumElements * NumBytesPerElement, TEXT("Lumen.PageBuffer"));
		FrameTemporaries.CardPageBufferSRV = GraphBuilder.CreateSRV(CardPageBufferRDG);
		FrameTemporaries.CardPageBufferUAV = GraphBuilder.CreateUAV(CardPageBufferRDG);

		if (NumElementsToUpload > 0)
		{
			UploadBuilder.AddPass(
				GraphBuilder,
				CardPageUploadBuffer,
				CardPageBufferRDG,
				NumElementsToUpload,
				NumBytesPerElement,
				TEXT("Lumen.CardPageUploadBuffer"),
				[this] (FRDGScatterUploader& Uploader)
			{
				const FVector2D InvPhysicalAtlasSize = FVector2D(1.0f) / GetPhysicalAtlasSize();

				FLumenPageTableEntry NullPageTableEntry;

				for (int32 PageIndex : PageTableIndicesToUpdateInBuffer)
				{
					if (PageIndex < PageTable.Num())
					{
						uint32 ResLevelPageTableOffset = 0;
						FIntPoint ResLevelSizeInTiles = FIntPoint(0, 0);

						FVector4f* Data = (FVector4f*)Uploader.Add_GetRef(PageIndex);

						if (PageTable.IsAllocated(PageIndex) && PageTable[PageIndex].IsMapped())
						{
							const FLumenPageTableEntry& PageTableEntry = PageTable[PageIndex];
							const FLumenCard& Card = Cards[PageTableEntry.CardIndex];
							const FLumenSurfaceMipMap& MipMap = Card.GetMipMap(PageTableEntry.ResLevel);

							ResLevelPageTableOffset = MipMap.PageTableSpanOffset;
							ResLevelSizeInTiles = MipMap.GetSizeInPages() * (Lumen::PhysicalPageSize / Lumen::CardTileSize);

							if (PageTableEntry.IsSubAllocation())
							{
								ResLevelSizeInTiles = PageTableEntry.SubAllocationSize / Lumen::CardTileSize;
							}

							FLumenCardPageGPUData::FillData(PageTableEntry, ResLevelPageTableOffset, ResLevelSizeInTiles, InvPhysicalAtlasSize, Data);
						}
						else
						{
							FLumenCardPageGPUData::FillData(NullPageTableEntry, ResLevelPageTableOffset, ResLevelSizeInTiles, InvPhysicalAtlasSize, Data);
						}
					}
				}
			});
		}

		// Resize also the CardPageLastUsedBuffers
		if (CardPageBufferOld != CardPageBuffer)
		{
			FRDGBufferRef CardPageLastUsedBufferRDG = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), NumElements), TEXT("Lumen.CardPageLastUsedBuffer"));

			FRDGBufferRef CardPageHighResLastUsedBufferRDG = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), NumElements), TEXT("Lumen.CardPageHighResLastUsedBuffer"));

			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(CardPageLastUsedBufferRDG), 0);
			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(CardPageHighResLastUsedBufferRDG), 0);

			CardPageLastUsedBuffer = GraphBuilder.ConvertToExternalBuffer(CardPageLastUsedBufferRDG);
			CardPageHighResLastUsedBuffer = GraphBuilder.ConvertToExternalBuffer(CardPageHighResLastUsedBufferRDG);
		}
	}
}

FLumenSceneData::FLumenSceneData(EShaderPlatform ShaderPlatform, EWorldType::Type WorldType, bool bAllowTrackAllPrimitivesInPreviewWorld) :
	bFinalLightingAtlasContentsValid(false)
{
	static const auto MeshCardCVar = IConsoleManager::Get().FindTConsoleVariableDataInt(TEXT("r.MeshCardRepresentation"));

	bTrackAllPrimitives = (DoesPlatformSupportLumenGI(ShaderPlatform)) 
		&& MeshCardCVar->GetValueOnGameThread() != 0
		&& ((WorldType != EWorldType::EditorPreview) || bAllowTrackAllPrimitivesInPreviewWorld);
}

FLumenSceneData::FLumenSceneData(bool bInTrackAllPrimitives) :
	bFinalLightingAtlasContentsValid(false)
{
	bTrackAllPrimitives = bInTrackAllPrimitives;
}

FLumenSceneData::~FLumenSceneData()
{
	LLM_SCOPE_BYTAG(Lumen);

	for (int32 CardIndex = 0; CardIndex < Cards.Num(); ++CardIndex)
	{
		if (Cards.IsAllocated(CardIndex))
		{
			RemoveCardFromAtlas(CardIndex);
		}
	}

	Cards.Reset();
	CardCullingInfos.Reset();
	MeshCards.Reset();
}

bool TrackPrimitiveInstanceForLumenScene(const FMatrix& LocalToWorld, const FBox& LocalBoundingBox, bool bEmissiveLightSource)
{
	const FVector LocalToWorldScale = LocalToWorld.GetScaleVector();
	const FVector ScaledBoundSize = LocalBoundingBox.GetSize() * LocalToWorldScale;
	const FVector FaceSurfaceArea(ScaledBoundSize.Y * ScaledBoundSize.Z, ScaledBoundSize.X * ScaledBoundSize.Z, ScaledBoundSize.Y * ScaledBoundSize.X);
	const float LargestFaceArea = FaceSurfaceArea.GetMax();

	const float MinFaceSurfaceArea = LumenMeshCards::GetCardMinSurfaceArea(bEmissiveLightSource);
	return LargestFaceArea > MinFaceSurfaceArea;
}

// Add function is a member of FScene, because it needs to add the primitive to all FLumenSceneData at once
void FScene::LumenAddPrimitive(FPrimitiveSceneInfo* InPrimitive)
{
	LLM_SCOPE_BYTAG(Lumen);

	if (DefaultLumenSceneData->bTrackAllPrimitives)
	{
		const FPrimitiveSceneProxy* Proxy = InPrimitive->Proxy;

		for (FLumenSceneDataIterator LumenSceneData = GetLumenSceneDataIterator(); LumenSceneData; ++LumenSceneData)
		{
			// We copy this flag over when creating per-view lumen scene data, validate that it's still the same
			check(LumenSceneData->bTrackAllPrimitives == DefaultLumenSceneData->bTrackAllPrimitives);

			LumenSceneData->PrimitivesToUpdateMeshCards.Add(InPrimitive->GetIndex());

			if (Proxy->IsVisibleInLumenScene())
			{
				ensure(!LumenSceneData->PendingAddOperations.Contains(InPrimitive));
				ensure(!LumenSceneData->PendingUpdateOperations.Contains(InPrimitive));
				ensure(InPrimitive->LumenPrimitiveGroupIndices.Num() == 0);
				LumenSceneData->PendingAddOperations.Add(InPrimitive);
			}
		}

		if (Proxy->IsLandscapeNaniteProxy())
		{
			for (FPrimitiveComponentId SourceComponentId : Proxy->GetSourceLandscapeComponentIds())
			{
				if (SourceComponentId.IsValid())
				{
					LandscapeToNaniteProxyMap.Add(SourceComponentId, InPrimitive);
				}
			}
		}
	}
}

// Update function is a member of FScene, because it needs to update the primitive for all FLumenSceneData at once
void FScene::LumenUpdatePrimitive(FPrimitiveSceneInfo* InPrimitive)
{
	LLM_SCOPE_BYTAG(Lumen);

	if (DefaultLumenSceneData->bTrackAllPrimitives
		&& InPrimitive->Proxy->IsVisibleInLumenScene()
		&& InPrimitive->LumenPrimitiveGroupIndices.Num() > 0)
	{
		for (FLumenSceneDataIterator LumenSceneData = GetLumenSceneDataIterator(); LumenSceneData; ++LumenSceneData)
		{
			if (!LumenSceneData->PendingUpdateOperations.Contains(InPrimitive)
				&& !LumenSceneData->PendingAddOperations.Contains(InPrimitive))
			{
				LumenSceneData->PendingUpdateOperations.Add(InPrimitive);
			}
		}
	}
}

// Update function is a member of FScene, because it needs to update the primitive for all FLumenSceneData at once
void FScene::LumenInvalidateSurfaceCacheForPrimitive(FPrimitiveSceneInfo* InPrimitive)
{
	LLM_SCOPE_BYTAG(Lumen);

	if (DefaultLumenSceneData->bTrackAllPrimitives
		&& InPrimitive->Proxy->IsVisibleInLumenScene()
		&& InPrimitive->LumenPrimitiveGroupIndices.Num() > 0)
	{
		for (FLumenSceneDataIterator LumenSceneData = GetLumenSceneDataIterator(); LumenSceneData; ++LumenSceneData)
		{
			if (!LumenSceneData->PendingSurfaceCacheInvalidationOperations.Contains(InPrimitive))
			{
				LumenSceneData->PendingSurfaceCacheInvalidationOperations.Add(InPrimitive);
			}
		}
	}
}

// Remove function is a member of FScene, because it needs to remove the primitive from all FLumenSceneData at once, mainly due to
// the FLumenPrimitiveGroupRemoveInfo constructor needing access to LumenPrimitiveGroupIndices from FPrimitiveSceneInfo, before it
// gets reset.
void FScene::LumenRemovePrimitive(FPrimitiveSceneInfo* InPrimitive, int32 PrimitiveIndex)
{
	LLM_SCOPE_BYTAG(Lumen);

	const FPrimitiveSceneProxy* Proxy = InPrimitive->Proxy;

	if (DefaultLumenSceneData->bTrackAllPrimitives
		&& InPrimitive->Proxy->IsVisibleInLumenScene())
	{
		for (FLumenSceneDataIterator LumenSceneData = GetLumenSceneDataIterator(); LumenSceneData; ++LumenSceneData)
		{
			LumenSceneData->PendingAddOperations.Remove(InPrimitive);
			LumenSceneData->PendingUpdateOperations.Remove(InPrimitive);
			LumenSceneData->PendingSurfaceCacheInvalidationOperations.Remove(InPrimitive);
			LumenSceneData->PendingRemoveOperations.Add(FLumenPrimitiveGroupRemoveInfo(InPrimitive, PrimitiveIndex));
		}

		InPrimitive->LumenPrimitiveGroupIndices.Reset();
	}

	if (DefaultLumenSceneData->bTrackAllPrimitives && Proxy->IsLandscapeNaniteProxy())
	{
		for (FPrimitiveComponentId SourceComponentId : Proxy->GetSourceLandscapeComponentIds())
		{
			LandscapeToNaniteProxyMap.Remove(SourceComponentId);
		}

#if DO_CHECK
		for (const auto& Pair : LandscapeToNaniteProxyMap)
		{
			check(Pair.Value != InPrimitive);
		}
#endif
	}
}

FLumenSceneDataIterator::FLumenSceneDataIterator(const FScene* InScene)
	: NextSceneData(InScene->PerViewOrGPULumenSceneData)
{
	Scene = InScene;
	LumenSceneData = nullptr;

	if (InScene->DefaultLumenSceneData)
	{
		LumenSceneData = InScene->DefaultLumenSceneData;
	}
	else
	{
		// Call increment operator to search for GPU or view specific Lumen scene data
		operator++();
	}
}

FLumenSceneDataIterator& FLumenSceneDataIterator::operator++()
{
	if (NextSceneData)
	{
		LumenSceneData = NextSceneData->Value;
		++NextSceneData;
	}
	else
	{
		LumenSceneData = nullptr;
	}
	return *this;
}

void FLumenSceneData::ResetAndConsolidate()
{
	// Reset arrays, but keep allocated memory for 1024 elements
	PendingAddOperations.Reset();
	PendingRemoveOperations.Reset(1024);
	PendingUpdateOperations.Reset();
	PendingUpdateOperations.Reserve(1024);
	PendingSurfaceCacheInvalidationOperations.Reset();
	PendingSurfaceCacheInvalidationOperations.Reserve(64);

	// Batch consolidate SparseSpanArrays
	PrimitiveGroups.Shrink();
	PrimitiveCullingInfos.Shrink();
	InstanceCullingInfos.Consolidate();
	Heightfields.Consolidate();
	MeshCards.Consolidate();
	Cards.Consolidate();
	CardCullingInfos.Consolidate();
	PageTable.Consolidate();
}

void FLumenSceneData::UpdatePrimitiveInstanceOffset(int32 PrimitiveIndex)
{
	if (bTrackAllPrimitives)
	{
		PrimitivesToUpdateMeshCards.Add(PrimitiveIndex);
	}
}

void UpdateLumenScenePrimitives(FRHIGPUMask GPUMask, FScene* Scene)
{
	LLM_SCOPE_BYTAG(Lumen);
	TRACE_CPUPROFILER_EVENT_SCOPE(UpdateLumenScenePrimitives);
	QUICK_SCOPE_CYCLE_COUNTER(UpdateLumenScenePrimitives);

	// Remove primitives
	for (FLumenSceneDataIterator LumenSceneData = Scene->GetLumenSceneDataIterator(); LumenSceneData; ++LumenSceneData)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE(RemoveLumenPrimitives);
		QUICK_SCOPE_CYCLE_COUNTER(RemoveLumenPrimitives);

		TSparseUniqueList<int32, SceneRenderingAllocator> PrimitiveGroupsToRemove;

		// Delete primitives
		for (const FLumenPrimitiveGroupRemoveInfo& RemoveInfo : LumenSceneData->PendingRemoveOperations)
		{
			for (int32 PrimitiveGroupIndex : RemoveInfo.LumenPrimitiveGroupIndices)
			{
				if (LumenSceneData->PrimitiveGroups.IsAllocated(PrimitiveGroupIndex))
				{
					FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndex];
					bool bPrimitiveRemoved = false;

					if (PrimitiveGroup.IsLandscape())
					{
						LumenSceneData->LandscapePrimitives.RemoveSingleSwap(RemoveInfo.Primitive);
					}

					for (int32 PrimitiveIndex = 0; PrimitiveIndex < PrimitiveGroup.GetPrimitives().Num(); ++PrimitiveIndex)
					{
						if (PrimitiveGroup.GetPrimitives()[PrimitiveIndex] == RemoveInfo.Primitive)
						{
							PrimitiveGroup.RemovePrimitiveAtSwap(PrimitiveIndex, EAllowShrinking::No);
							bPrimitiveRemoved = true;
							break;
						}
					}

					// FPrimitiveSceneInfo::LumenPrimitiveGroupIndices may have primitive group indices from different Lumen scenes
					// (e.g. a default one and another one for split screen) so make sure this primitive group actually contain the
					// primitive being removed before removing the group
					if (bPrimitiveRemoved)
					{
						if (PrimitiveGroup.RayTracingGroupMapElementId.IsValid())
						{
							// Removal of multi-component primitive groups are deferred because they may not be empty after the removal
							PrimitiveGroupsToRemove.Add(PrimitiveGroupIndex);
						}
						else
						{
							check(PrimitiveGroup.GetPrimitives().Num() == 0);
							LumenSceneData->RemoveMeshCards(PrimitiveGroupIndex, /*bUpdateCullingInfo*/ false);
							LumenSceneData->RemovePrimitiveGroupCullingInfo(PrimitiveGroup);
							LumenSceneData->PrimitiveGroups.RemoveAt(PrimitiveGroupIndex);
							LumenSceneData->UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);
						}
					}
				}
			}
		}

		// Update multi-component primitive groups or remove them if they are empty
		for (int32 PrimitiveGroupIndex : PrimitiveGroupsToRemove.Array)
		{
			FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndex];
			check(PrimitiveGroup.RayTracingGroupMapElementId.IsValid());

			LumenSceneData->RemoveMeshCards(PrimitiveGroupIndex, /*bUpdateCullingInfo*/ false);

			if (PrimitiveGroup.GetPrimitives().Num() == 0)
			{
				LumenSceneData->RayTracingGroups.RemoveByElementId(PrimitiveGroup.RayTracingGroupMapElementId);
				PrimitiveGroup.RayTracingGroupMapElementId = Experimental::FHashElementId();
				LumenSceneData->RemovePrimitiveGroupCullingInfo(PrimitiveGroup);
				LumenSceneData->PrimitiveGroups.RemoveAt(PrimitiveGroupIndex);
			}
			else
			{
				// Update bounds
				FBox WorldSpaceBoundingBox;
				WorldSpaceBoundingBox.Init();
				for (const FPrimitiveSceneInfo* Primitive : PrimitiveGroup.GetPrimitives())
				{
					WorldSpaceBoundingBox += Primitive->Proxy->GetBounds().GetBox();
				}
				LumenSceneData->UpdatePrimitiveGroupCullingInfo(PrimitiveGroup, WorldSpaceBoundingBox);
			}

			LumenSceneData->UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);
		}
	}

	// Add primitives
	for (FLumenSceneDataIterator LumenSceneData = Scene->GetLumenSceneDataIterator(); LumenSceneData; ++LumenSceneData)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE(AddLumenPrimitives);
		QUICK_SCOPE_CYCLE_COUNTER(AddLumenPrimitives);

		for (FPrimitiveSceneInfo* ScenePrimitiveInfo : LumenSceneData->PendingAddOperations)
		{
			FPrimitiveSceneProxy* SceneProxy = ScenePrimitiveInfo->Proxy;

			const int32 NumInstances = ScenePrimitiveInfo->GetNumInstanceSceneDataEntries();
			const FInstanceSceneDataBuffers *InstanceData = ScenePrimitiveInfo->GetInstanceSceneDataBuffers();

			// Instance data must be available on CPU.
			check(!InstanceData || !InstanceData->IsInstanceDataGPUOnly());

			bool bAnyInstanceValid = false;
			{
				const FMatrix& PrimitiveToWorld = SceneProxy->GetLocalToWorld();

				for (int32 InstanceIndex = 0; InstanceIndex < NumInstances; ++InstanceIndex)
				{
					FBox LocalBoundingBox = SceneProxy->GetLocalBounds().GetBox();
					FMatrix LocalToWorld = PrimitiveToWorld;

					if (InstanceData)
					{
						LocalToWorld = InstanceData->GetInstanceToWorld(InstanceIndex);
						LocalBoundingBox = InstanceData->GetInstanceLocalBounds(InstanceIndex).ToBox();
					}

					if (TrackPrimitiveInstanceForLumenScene(LocalToWorld, LocalBoundingBox, SceneProxy->IsEmissiveLightSource()))
					{
						bAnyInstanceValid = true;
						break;
					}
				}
			}

			LumenSceneData->PrimitivesToUpdateMeshCards.Add(ScenePrimitiveInfo->GetIndex());

			if (bAnyInstanceValid)
			{
				auto GetCustomId = [ScenePrimitiveInfo, SceneProxy]()
				{
					uint32 CustomId = UINT32_MAX;
					if (SceneProxy->IsCompatibleWithLumenCardSharing())
					{
						FXxHash64Builder Hasher;

						check(SceneProxy->IsNaniteMesh());

						uint32 RuntimeResourceID = static_cast<const Nanite::FSceneProxyBase*>(SceneProxy)->GetResourcePrimitiveInfo().ResourceID;
						check(RuntimeResourceID != INDEX_NONE);

						Hasher.Update(&RuntimeResourceID, sizeof(RuntimeResourceID));
						for (const FNaniteShadingBin& ShadingBin : ScenePrimitiveInfo->NaniteShadingBins[ENaniteMeshPass::LumenCardCapture])
						{
							Hasher.Update(&ShadingBin, sizeof(ShadingBin));
						}

						const FXxHash64 Hash = Hasher.Finalize();
						CustomId = GetTypeHash(Hash);
					}
					return CustomId;
				};

				// First try to merge components
				extern int32 GLumenMeshCardsMergeComponents;
				if (GLumenMeshCardsMergeComponents != 0 
					&& SceneProxy->GetRayTracingGroupId() != FPrimitiveSceneProxy::InvalidRayTracingGroupId
					&& !SceneProxy->IsEmissiveLightSource()
					&& SceneProxy->IsOpaqueOrMasked())
				{
					const Experimental::FHashElementId RayTracingGroupMapElementId = LumenSceneData->RayTracingGroups.FindOrAddId(SceneProxy->GetRayTracingGroupId(), -1);
					int32& PrimitiveGroupIndex = LumenSceneData->RayTracingGroups.GetByElementId(RayTracingGroupMapElementId).Value;

					if (PrimitiveGroupIndex >= 0)
					{
						ScenePrimitiveInfo->LumenPrimitiveGroupIndices.Add(PrimitiveGroupIndex);

						FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndex];
						ensure(PrimitiveGroup.RayTracingGroupMapElementId == RayTracingGroupMapElementId && PrimitiveGroup.CustomId == UINT32_MAX);

						LumenSceneData->RemoveMeshCards(PrimitiveGroupIndex, /*bUpdateCullingInfo*/ false);
						PrimitiveGroup.SetHasValidMeshCards(true);
						PrimitiveGroup.AddPrimitive(ScenePrimitiveInfo);

						LumenSceneData->UpdatePrimitiveGroupCullingInfo(PrimitiveGroup, SceneProxy->GetBounds().GetBox(), false /*bForcePrimitiveLevel*/, true /*bAdditive*/);
					}
					else
					{
						PrimitiveGroupIndex = LumenSceneData->PrimitiveGroups.AddDefaulted();
						ScenePrimitiveInfo->LumenPrimitiveGroupIndices.Add(PrimitiveGroupIndex);
						LumenSceneData->UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);

						FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndex];
						PrimitiveGroup.RayTracingGroupMapElementId = RayTracingGroupMapElementId;
						PrimitiveGroup.PrimitiveInstanceIndex = -1;
						PrimitiveGroup.CardResolutionScale = 1.0f;
						PrimitiveGroup.MeshCardsIndex = -1;
						PrimitiveGroup.SetHasValidMeshCards(true);
						PrimitiveGroup.SetIsFarField(SceneProxy->IsRayTracingFarField());
						PrimitiveGroup.SetIsHeightfield(false);
						PrimitiveGroup.SetIsOpaqueOrMasked(true);
						PrimitiveGroup.SetLightingChannelMask(SceneProxy->GetLightingChannelMask());
						PrimitiveGroup.AddPrimitive(ScenePrimitiveInfo);

						const FRenderBounds WorldSpaceBoundingBox = SceneProxy->GetBounds().GetBox();
						const FSparseArrayAllocationInfo AllocInfo = LumenSceneData->PrimitiveCullingInfos.AddUninitialized();
						PrimitiveGroup.PrimitiveCullingInfoIndex = AllocInfo.Index;
						new (AllocInfo.Pointer) FLumenPrimitiveGroupCullingInfo(WorldSpaceBoundingBox, PrimitiveGroup, PrimitiveGroupIndex);
					}
				}
				else
				{
					const FMatrix& LocalToWorld = SceneProxy->GetLocalToWorld();

					bool bMergedInstances = false;

					if (NumInstances > 1)
					{
						// Check if we can merge all instances into one MeshCards
						extern int32 GLumenMeshCardsMergeInstances;
						extern float GLumenMeshCardsMergedMaxWorldSize;

						const FBox PrimitiveBox = SceneProxy->GetBounds().GetBox();
						const FRenderBounds PrimitiveBounds = FRenderBounds(PrimitiveBox);

						if (GLumenMeshCardsMergeInstances
							&& NumInstances > 1
							&& PrimitiveBox.GetSize().GetMax() < GLumenMeshCardsMergedMaxWorldSize)
						{
							FRenderBounds PrimitiveRelativeBounds;
							double TotalInstanceSurfaceArea = 0;

							for (int32 InstanceIndex = 0; InstanceIndex < NumInstances; ++InstanceIndex)
							{
								const FRenderBounds InstanceBounds = InstanceData->GetInstancePrimitiveRelativeBounds(InstanceIndex);
								PrimitiveRelativeBounds += InstanceBounds;
								const double InstanceSurfaceArea = BoxSurfaceArea((FVector)InstanceBounds.GetExtent());
								TotalInstanceSurfaceArea += InstanceSurfaceArea;
							}

							const double BoundsSurfaceArea = BoxSurfaceArea((FVector)PrimitiveRelativeBounds.GetExtent());
							const float SurfaceAreaRatio = BoundsSurfaceArea / TotalInstanceSurfaceArea;

							extern float GLumenMeshCardsMergeInstancesMaxSurfaceAreaRatio;
							extern float GLumenMeshCardsMergedResolutionScale;

							if (SurfaceAreaRatio < GLumenMeshCardsMergeInstancesMaxSurfaceAreaRatio)
							{
								const int32 PrimitiveGroupIndex = LumenSceneData->PrimitiveGroups.AddDefaulted();
								ScenePrimitiveInfo->LumenPrimitiveGroupIndices.Add(PrimitiveGroupIndex);
								LumenSceneData->UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);

								FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndex];
								PrimitiveGroup.PrimitiveInstanceIndex = -1;
								PrimitiveGroup.CardResolutionScale = FMath::Sqrt(1.0f / SurfaceAreaRatio) * GLumenMeshCardsMergedResolutionScale;
								PrimitiveGroup.MeshCardsIndex = -1;
								PrimitiveGroup.HeightfieldIndex = -1;
								PrimitiveGroup.SetHasValidMeshCards(true);
								PrimitiveGroup.SetIsFarField(SceneProxy->IsRayTracingFarField());
								PrimitiveGroup.SetIsHeightfield(false);
								PrimitiveGroup.SetLightingChannelMask(SceneProxy->GetLightingChannelMask());
								PrimitiveGroup.SetIsEmissiveLightSource(SceneProxy->IsEmissiveLightSource());
								PrimitiveGroup.SetIsOpaqueOrMasked(SceneProxy->IsOpaqueOrMasked());
								PrimitiveGroup.AddPrimitive(ScenePrimitiveInfo);

								const FRenderBounds WorldSpaceBoundingBox = PrimitiveRelativeBounds.ToBox().ShiftBy(InstanceData->GetPrimitiveWorldSpaceOffset());
								const FSparseArrayAllocationInfo AllocInfo = LumenSceneData->PrimitiveCullingInfos.AddUninitialized();
								PrimitiveGroup.PrimitiveCullingInfoIndex = AllocInfo.Index;
								new (AllocInfo.Pointer) FLumenPrimitiveGroupCullingInfo(WorldSpaceBoundingBox, PrimitiveGroup, PrimitiveGroupIndex);

								bMergedInstances = true;
							}

							#define LOG_LUMEN_PRIMITIVE_ADDS 0
							#if LOG_LUMEN_PRIMITIVE_ADDS
							{
								UE_LOGF(LogRenderer, Log, "AddLumenPrimitive %ls: Instances: %u, Merged: %u, SurfaceAreaRatio: %.1f",
									*LumenPrimitive.Primitive->Proxy->GetOwnerName().ToString(),
									NumInstances,
									LumenPrimitive.bMergedInstances ? 1 : 0,
									SurfaceAreaRatio);
							}
							#endif
						}

						if (!bMergedInstances)
						{
							const int32 InstanceCullingInfoOffset = LumenSceneData->InstanceCullingInfos.AddSpan(NumInstances);

							const FSparseArrayAllocationInfo AllocInfo = LumenSceneData->PrimitiveCullingInfos.AddUninitialized();
							const int32 PrimitiveCullingInfoIndex = AllocInfo.Index;
							FRenderBounds WorldSpaceBoundingBox;

							const uint32 PrimitiveGroupOffset = ScenePrimitiveInfo->LumenPrimitiveGroupIndices.AddDefaulted(NumInstances);
							const uint32 CustomId = GetCustomId();

							for (int32 InstanceIndex = 0; InstanceIndex < NumInstances; ++InstanceIndex)
							{
								const int32 PrimitiveGroupIndex = LumenSceneData->PrimitiveGroups.AddDefaulted();
								ScenePrimitiveInfo->LumenPrimitiveGroupIndices[PrimitiveGroupOffset + InstanceIndex] = PrimitiveGroupIndex;
								LumenSceneData->UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);

								FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndex];
								PrimitiveGroup.PrimitiveInstanceIndex = InstanceIndex;
								PrimitiveGroup.CardResolutionScale = 1.0f;
								PrimitiveGroup.MeshCardsIndex = -1;
								PrimitiveGroup.HeightfieldIndex = -1;
								PrimitiveGroup.PrimitiveCullingInfoIndex = PrimitiveCullingInfoIndex;
								PrimitiveGroup.InstanceCullingInfoIndex = InstanceCullingInfoOffset + InstanceIndex;
								PrimitiveGroup.SetHasValidMeshCards(true);
								PrimitiveGroup.SetIsFarField(SceneProxy->IsRayTracingFarField());
								PrimitiveGroup.SetIsHeightfield(false);
								PrimitiveGroup.SetLightingChannelMask(SceneProxy->GetLightingChannelMask());
								PrimitiveGroup.SetIsEmissiveLightSource(SceneProxy->IsEmissiveLightSource());
								PrimitiveGroup.SetIsOpaqueOrMasked(SceneProxy->IsOpaqueOrMasked());
								PrimitiveGroup.AddPrimitive(ScenePrimitiveInfo);
								PrimitiveGroup.CustomId = CustomId;

								const FRenderBounds InstanceBounds = InstanceData->GetInstanceWorldBounds(InstanceIndex);
								WorldSpaceBoundingBox += InstanceBounds;
								new (&LumenSceneData->InstanceCullingInfos[InstanceCullingInfoOffset + InstanceIndex]) FLumenPrimitiveGroupCullingInfo(InstanceBounds, PrimitiveGroup, PrimitiveGroupIndex);
							}

							new (AllocInfo.Pointer) FLumenPrimitiveGroupCullingInfo(WorldSpaceBoundingBox, InstanceCullingInfoOffset, NumInstances, SceneProxy->IsRayTracingFarField());
						}
					}
					else
					{
						const int32 PrimitiveGroupIndex = LumenSceneData->PrimitiveGroups.AddDefaulted();
						ScenePrimitiveInfo->LumenPrimitiveGroupIndices.Add(PrimitiveGroupIndex);
						LumenSceneData->UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);

						FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndex];
						PrimitiveGroup.PrimitiveInstanceIndex = 0;
						PrimitiveGroup.CardResolutionScale = 1.0f;
						PrimitiveGroup.MeshCardsIndex = -1;
						PrimitiveGroup.HeightfieldIndex = -1;
						PrimitiveGroup.SetHasValidMeshCards(true);
						PrimitiveGroup.SetIsFarField(SceneProxy->IsRayTracingFarField());
						PrimitiveGroup.SetIsHeightfield(SceneProxy->SupportsHeightfieldRepresentation() || SceneProxy->IsLumenHeightfield());
						PrimitiveGroup.SetLightingChannelMask(SceneProxy->GetLightingChannelMask());
						PrimitiveGroup.SetIsEmissiveLightSource(SceneProxy->IsEmissiveLightSource());
						PrimitiveGroup.SetIsOpaqueOrMasked(SceneProxy->IsOpaqueOrMasked());
						PrimitiveGroup.SetIsLandscape(SceneProxy->IsLandscapeProxy());
						PrimitiveGroup.AddPrimitive(ScenePrimitiveInfo);
						PrimitiveGroup.CustomId = GetCustomId();

						const FRenderBounds WorldSpaceBoundingBox = SceneProxy->GetBounds().GetBox();
						const FSparseArrayAllocationInfo AllocInfo = LumenSceneData->PrimitiveCullingInfos.AddUninitialized();
						PrimitiveGroup.PrimitiveCullingInfoIndex = AllocInfo.Index;
						new (AllocInfo.Pointer) FLumenPrimitiveGroupCullingInfo(WorldSpaceBoundingBox, PrimitiveGroup, PrimitiveGroupIndex);

						if (PrimitiveGroup.IsLandscape())
						{
							LumenSceneData->LandscapePrimitives.Add(ScenePrimitiveInfo);
						}
					}
				}
			}
		}
	}

	// UpdateLumenPrimitives
	for (FLumenSceneDataIterator LumenSceneData = Scene->GetLumenSceneDataIterator(); LumenSceneData; ++LumenSceneData)
	{
		QUICK_SCOPE_CYCLE_COUNTER(UpdateLumenPrimitives);

		for (TSet<FPrimitiveSceneInfo*>::TIterator It(LumenSceneData->PendingUpdateOperations); It; ++It)
		{
			FPrimitiveSceneInfo* PrimitiveSceneInfo = *It;

			if (PrimitiveSceneInfo->LumenPrimitiveGroupIndices.Num() > 0)
			{
				const FCardRepresentationData* CardRepresentationData = PrimitiveSceneInfo->Proxy->GetMeshCardRepresentation();
				const FMatrix& PrimitiveToWorld = PrimitiveSceneInfo->Proxy->GetLocalToWorld();

				const FInstanceSceneDataBuffers *InstanceData = PrimitiveSceneInfo->GetInstanceSceneDataBuffers();
				const int32 NumInstances = PrimitiveSceneInfo->GetNumInstanceSceneDataEntries();
				int32 PrimitiveGroupIndexIfUpdatePrimitiveBounds = INDEX_NONE;
				int32 PrimitiveCullingInfoIndex = INDEX_NONE;
				FRenderBounds PrimitiveWorldBounds;

				// Instance data must be available on CPU.
				check(!InstanceData || !InstanceData->IsInstanceDataGPUOnly());

				for (int32 PrimitiveGroupIndex : PrimitiveSceneInfo->LumenPrimitiveGroupIndices)
				{
					if (LumenSceneData->PrimitiveGroups.IsAllocated(PrimitiveGroupIndex))
					{
						FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndex];

						if (PrimitiveGroup.PrimitiveInstanceIndex >= 0 && PrimitiveGroup.GetPrimitives()[0] == PrimitiveSceneInfo)
						{
							check(PrimitiveGroup.GetPrimitives().Num() == 1);

							FBox WorldSpaceBoundingBox;
							if (InstanceData)
							{
								WorldSpaceBoundingBox = InstanceData->GetInstanceWorldBounds(PrimitiveGroup.PrimitiveInstanceIndex).GetBox();

								if (NumInstances > 1)
								{
									check(PrimitiveCullingInfoIndex == INDEX_NONE || PrimitiveCullingInfoIndex == PrimitiveGroup.PrimitiveCullingInfoIndex);

									PrimitiveGroupIndexIfUpdatePrimitiveBounds = PrimitiveGroupIndex;
									PrimitiveCullingInfoIndex = PrimitiveGroup.PrimitiveCullingInfoIndex;
									PrimitiveWorldBounds += WorldSpaceBoundingBox;
								}
							}
							else
							{
								WorldSpaceBoundingBox = PrimitiveSceneInfo->Proxy->GetBounds().GetBox();
							}

							const FMatrix LocalToWorld = PrimitiveSceneInfo->Proxy->GetMeshCardsToWorld(PrimitiveGroup.PrimitiveInstanceIndex);

							LumenSceneData->UpdateMeshCards(LocalToWorld, PrimitiveGroup.MeshCardsIndex, CardRepresentationData->MeshCardsBuildData);
							LumenSceneData->UpdatePrimitiveGroupCullingInfo(PrimitiveGroup, WorldSpaceBoundingBox);

							LumenSceneData->UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);
						}
					}
				}

				if (PrimitiveGroupIndexIfUpdatePrimitiveBounds >= 0)
				{
					const FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndexIfUpdatePrimitiveBounds];
					check(PrimitiveCullingInfoIndex == PrimitiveGroup.PrimitiveCullingInfoIndex);
					LumenSceneData->UpdatePrimitiveGroupCullingInfo(PrimitiveGroup, PrimitiveWorldBounds, /*bForcePrimitiveLevel*/ true);
				}
			}
		}
	}

	// InvalidateLumenSurfaceCacheForPrimitives
	for (FLumenSceneDataIterator LumenSceneData = Scene->GetLumenSceneDataIterator(); LumenSceneData; ++LumenSceneData)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE(InvalidateLumenSurfaceCacheForPrimitives);
		QUICK_SCOPE_CYCLE_COUNTER(InvalidateLumenSurfaceCacheForPrimitives);

		for (TSet<FPrimitiveSceneInfo*>::TIterator It(LumenSceneData->PendingSurfaceCacheInvalidationOperations); It; ++It)
		{
			FPrimitiveSceneInfo* PrimitiveSceneInfo = *It;

			if (PrimitiveSceneInfo->LumenPrimitiveGroupIndices.Num() > 0)
			{
				const FCardRepresentationData* CardRepresentationData = PrimitiveSceneInfo->Proxy->GetMeshCardRepresentation();
				const FMatrix& PrimitiveToWorld = PrimitiveSceneInfo->Proxy->GetLocalToWorld();

				for (int32 PrimitiveGroupIndex : PrimitiveSceneInfo->LumenPrimitiveGroupIndices)
				{
					const FLumenPrimitiveGroup& PrimitiveGroup = LumenSceneData->PrimitiveGroups[PrimitiveGroupIndex];
					if (PrimitiveGroup.MeshCardsIndex >= 0)
					{
						LumenSceneData->InvalidateSurfaceCache(GPUMask, PrimitiveGroup.MeshCardsIndex);
					}
				}
			}
		}
	}

	for (FLumenSceneDataIterator LumenSceneData = Scene->GetLumenSceneDataIterator(); LumenSceneData; ++LumenSceneData)
	{
		LumenSceneData->ResetAndConsolidate();
	}
}

void FLumenSceneData::ReleaseAtlas()
{
	RemoveAllMeshCards();

	PhysicalAtlasSize = 0;

	AlbedoAtlas.SafeRelease();
	NormalAtlas.SafeRelease();
	EmissiveAtlas.SafeRelease();
	DepthAtlas.SafeRelease();

	DirectLightingAtlas.SafeRelease();
	IndirectLightingAtlas.SafeRelease();
	RadiosityNumFramesAccumulatedAtlas.SafeRelease();
	FinalLightingAtlas.SafeRelease();
	TileShadowDownsampleFactorAtlas.SafeRelease();
	DiffuseLightingAndSecondMomentHistoryAtlas.SafeRelease();
	NumFramesAccumulatedHistoryAtlas.SafeRelease();

	RadiosityTraceRadianceAtlas.SafeRelease();
	RadiosityTraceHitDistanceAtlas.SafeRelease();
	RadiosityProbeSHRedAtlas.SafeRelease();
	RadiosityProbeSHGreenAtlas.SafeRelease();
	RadiosityProbeSHBlueAtlas.SafeRelease();
}

void FLumenSceneData::RemoveAllMeshCards()
{
	LLM_SCOPE_BYTAG(Lumen);
	QUICK_SCOPE_CYCLE_COUNTER(RemoveAllCards);

	for (int32 PrimitiveGroupIndex = 0; PrimitiveGroupIndex < PrimitiveGroups.GetMaxSize(); ++PrimitiveGroupIndex)
	{
		if (PrimitiveGroups.IsAllocated(PrimitiveGroupIndex))
		{
			RemoveMeshCards(PrimitiveGroupIndex);
		}
	}

	CardSharingInfoMap.Empty();
	PendingRemoveCardSharingInfos.Empty();
	PendingAddCardSharingInfos.Empty();
}

bool FLumenSceneData::UpdateAtlasSize()
{
	const ESurfaceCacheCompression NewCompression = GetSurfaceCacheCompression();

	if (PhysicalAtlasSize != GetDesiredPhysicalAtlasSize(SurfaceCacheResolution)
		|| PhysicalAtlasCompression != NewCompression
		|| CurrentLightingDataFormat != Lumen::GetLightingDataFormat()
		|| CurrentCachedLightingPreExposure != Lumen::GetCachedLightingPreExposure())
	{
		RemoveAllMeshCards();

		PhysicalAtlasSize = GetDesiredPhysicalAtlasSize(SurfaceCacheResolution);
		SurfaceCacheAllocator.Init(GetDesiredPhysicalAtlasSizeInPages(SurfaceCacheResolution));
		PhysicalAtlasCompression = NewCompression;
		CurrentLightingDataFormat = Lumen::GetLightingDataFormat();
		CurrentCachedLightingPreExposure = Lumen::GetCachedLightingPreExposure();

		return true;
	}

	return false;
}

void FLumenSceneData::MapSurfaceCachePage(const FLumenSurfaceMipMap& MipMap, int32 PageTableIndex, FRHIGPUMask GPUMask)
{
	FLumenPageTableEntry& PageTableEntry = PageTable[PageTableIndex];
	if (!PageTableEntry.IsMapped())
	{
		FLumenSurfaceCacheAllocator::FAllocation Allocation;
		SurfaceCacheAllocator.Allocate(PageTableEntry, Allocation);

		PageTableEntry.PhysicalPageCoord = Allocation.PhysicalPageCoord;
		PageTableEntry.PhysicalAtlasRect = Allocation.PhysicalAtlasRect;

		if (PageTableEntry.IsMapped())
		{
			PageTableEntry.SamplePageIndex = PageTableIndex;
			PageTableEntry.SampleAtlasBiasX = PageTableEntry.PhysicalAtlasRect.Min.X / Lumen::MinCardResolution;
			PageTableEntry.SampleAtlasBiasY = PageTableEntry.PhysicalAtlasRect.Min.Y / Lumen::MinCardResolution;
			PageTableEntry.SampleCardResLevelX = MipMap.ResLevelX;
			PageTableEntry.SampleCardResLevelY = MipMap.ResLevelY;

			for (uint32 GPUIndex = 0; GPUIndex < GNumExplicitGPUsForRendering; GPUIndex++)
			{
				LastCapturedPageHeap[GPUIndex].Add(
#if WITH_MGPU
					GPUMask.Contains(GPUIndex) ? GetSurfaceCacheUpdateFrameIndex() : 0,
#else
					GetSurfaceCacheUpdateFrameIndex(),
#endif
					PageTableIndex);
			}

			if (!MipMap.bLocked)
			{
				UnlockedAllocationHeap.Add(SurfaceCacheFeedback.GetFrameIndex(), PageTableIndex);
			}
		}

		PageTableIndicesToUpdateInBuffer.Add(PageTableIndex);
	}
}

void FLumenSceneData::UnmapSurfaceCachePage(bool bLocked, FLumenPageTableEntry& Page, int32 PageIndex)
{
	if (Page.IsMapped())
	{
		for (uint32 GPUIndex = 0; GPUIndex < GNumExplicitGPUsForRendering; GPUIndex++)
		{
			LastCapturedPageHeap[GPUIndex].Remove(PageIndex);
			PagesToRecaptureHeap[GPUIndex].Remove(PageIndex);
		}

		if (!bLocked)
		{
			UnlockedAllocationHeap.Remove(PageIndex);
		}

		SurfaceCacheAllocator.Free(Page);

		Page.PhysicalPageCoord.X = -1;
		Page.PhysicalPageCoord.Y = -1;
		Page.SampleAtlasBiasX = 0;
		Page.SampleAtlasBiasY = 0;
		Page.SampleCardResLevelX = 0;
		Page.SampleCardResLevelY = 0;
	}
}

const FLumenCardSharingInfo* FLumenSceneData::FindMatchingCardForCopy(const FLumenCardId& CardId, uint32 ResLevel) const
{
	if (bAllowCardSharing && CardId.IsValid())
	{
		const TSparseArray<FLumenCardSharingInfo>* CardInfos = CardSharingInfoMap.Find(CardId);

		if (CardInfos)
		{
			for (const FLumenCardSharingInfo& CardInfo : *CardInfos)
			{
				if (ResLevel <= CardInfo.MinAllocatedResLevel)
				{
					return &CardInfo;
				}
			}
		}
	}

	return nullptr;
}

void FLumenSceneData::FlushPendingCardSharingInfos()
{
	if (!bAllowCardSharing)
	{
		return;
	}

	TArray<FLumenCardSharingInfoPendingRemove>& PendingRemoves = PendingRemoveCardSharingInfos;
	TArray<FLumenCardSharingInfoPendingAdd>& PendingAdds = PendingAddCardSharingInfos;

	Algo::Sort(PendingRemoves);
	Algo::Sort(PendingAdds);

	const int32 NumRemoves = PendingRemoves.Num();
	const int32 NumAdds = PendingAdds.Num();
	int32 RemoveIndex = 0;
	int32 AddIndex = 0;

	while (RemoveIndex < NumRemoves || AddIndex < NumAdds)
	{
		while (RemoveIndex < NumRemoves && (AddIndex >= NumAdds || PendingRemoves[RemoveIndex].CardId <= PendingAdds[AddIndex].CardId))
		{
			const FLumenCardSharingInfoPendingRemove& Info = PendingRemoves[RemoveIndex];
			const FSetElementId Id = CardSharingInfoMap.FindId(Info.CardId);
			TSparseArray<FLumenCardSharingInfo>& CardInfos = CardSharingInfoMap.Get(Id).Value;

			CardInfos.RemoveAt(Info.CardSharingListIndex);
			if (CardInfos.IsEmpty() && (AddIndex >= NumAdds || Info.CardId < PendingAdds[AddIndex].CardId))
			{
				CardSharingInfoMap.Remove(Id);
			}
			++RemoveIndex;
		}

		while (AddIndex < NumAdds && (RemoveIndex >= NumRemoves || PendingAdds[AddIndex].CardId < PendingRemoves[RemoveIndex].CardId))
		{
			const FLumenCardSharingInfoPendingAdd& Info = PendingAdds[AddIndex];
			TSparseArray<FLumenCardSharingInfo>& CardInfos = CardSharingInfoMap.FindOrAdd(Info.CardId);
			FLumenCard& Card = Cards[Info.CardIndex];

			check(Info.CardId == Card.CardSharingId && Info.MinAllocatedResLevel == Card.MinAllocatedResLevel && Info.bAxisXFlipped == Card.bAxisXFlipped); //-V1013
			Card.CardSharingListIndex = CardInfos.Add(FLumenCardSharingInfo(Info.CardIndex, Info.MinAllocatedResLevel, Info.bAxisXFlipped));
			++AddIndex;
		}
	}

	PendingRemoves.Reset();
	PendingAdds.Reset();
}

void FLumenSceneData::ReallocVirtualSurface(FLumenCard& Card, int32 CardIndex, int32 ResLevel, bool bLockPages)
{
	FLumenSurfaceMipMap& MipMap = Card.GetMipMap(ResLevel);

	if (MipMap.PageTableSpanSize > 0 && MipMap.bLocked != bLockPages)
	{
		// Virtual memory is already allocated, but need to change the bLocked flag for any mapped pages
	
		if (MipMap.bLocked)
		{
			// Unlock all pages
			for (int32 LocalPageIndex = 0; LocalPageIndex < int32(MipMap.SizeInPagesX * MipMap.SizeInPagesY); ++LocalPageIndex)
			{
				const int32 PageTableIndex = MipMap.GetPageTableIndex(LocalPageIndex);
				FLumenPageTableEntry& PageTableEntry = PageTable[PageTableIndex];
				if (PageTableEntry.IsMapped())
				{
					UnlockedAllocationHeap.Add(SurfaceCacheFeedback.GetFrameIndex(), PageTableIndex);
				}
			}

			MipMap.bLocked = false;
		}
		else
		{
			// Lock all pages
			for (int32 LocalPageIndex = 0; LocalPageIndex < int32(MipMap.SizeInPagesX * MipMap.SizeInPagesY); ++LocalPageIndex)
			{
				const int32 PageTableIndex = MipMap.GetPageTableIndex(LocalPageIndex);
				FLumenPageTableEntry& PageTableEntry = PageTable[PageTableIndex];
				if (PageTableEntry.IsMapped())
				{
					UnlockedAllocationHeap.Remove(PageTableIndex);
				}
			}

			MipMap.bLocked = true;
		}
	}
	else if (MipMap.PageTableSpanSize == 0)
	{
		// Allocate virtual memory for the given mip map

		FLumenMipMapDesc MipMapDesc;
		Card.GetMipMapDesc(ResLevel, MipMapDesc);

		MipMap.bLocked = bLockPages;
		MipMap.SizeInPagesX = MipMapDesc.SizeInPages.X;
		MipMap.SizeInPagesY = MipMapDesc.SizeInPages.Y;
		MipMap.ResLevelX = MipMapDesc.ResLevelX;
		MipMap.ResLevelY = MipMapDesc.ResLevelY;
		MipMap.PageTableSpanSize = MipMapDesc.SizeInPages.X * MipMapDesc.SizeInPages.Y;
		MipMap.PageTableSpanOffset = PageTable.AddSpan(MipMap.PageTableSpanSize);

		for (int32 LocalPageIndex = 0; LocalPageIndex < MipMapDesc.SizeInPages.X * MipMapDesc.SizeInPages.Y; ++LocalPageIndex)
		{
			const int32 PageTableIndex = MipMap.GetPageTableIndex(LocalPageIndex);

			FLumenPageTableEntry& PageTableEntry = PageTable[PageTableIndex];
			PageTableEntry.CardIndex = CardIndex;
			PageTableEntry.ResLevel = ResLevel;
			PageTableEntry.SubAllocationSize = MipMapDesc.bSubAllocation ? MipMapDesc.Resolution : FIntPoint(-1, -1);
			PageTableEntry.SampleAtlasBiasX = 0;
			PageTableEntry.SampleAtlasBiasY = 0;
			PageTableEntry.SampleCardResLevelX = 0;
			PageTableEntry.SampleCardResLevelY = 0;

			const int32 LocalPageCoordX = LocalPageIndex % MipMapDesc.SizeInPages.X;
			const int32 LocalPageCoordY = LocalPageIndex / MipMapDesc.SizeInPages.X;

			FVector4f CardUVRect;
			CardUVRect.X = float(LocalPageCoordX + 0.0f) / MipMapDesc.SizeInPages.X;
			CardUVRect.Y = float(LocalPageCoordY + 0.0f) / MipMapDesc.SizeInPages.Y;
			CardUVRect.Z = float(LocalPageCoordX + 1.0f) / MipMapDesc.SizeInPages.X;
			CardUVRect.W = float(LocalPageCoordY + 1.0f) / MipMapDesc.SizeInPages.Y;

			// Every page has a 0.5 texel border for correct bilinear sampling
			// This border is only needed on interior page edges
			{
				FVector2D CardBorderOffset;
				CardBorderOffset = FVector2D(0.5f * (Lumen::PhysicalPageSize - Lumen::VirtualPageSize));
				CardBorderOffset.X *= (CardUVRect.Z - CardUVRect.X) / Lumen::PhysicalPageSize;
				CardBorderOffset.Y *= (CardUVRect.W - CardUVRect.Y) / Lumen::PhysicalPageSize;

				if (LocalPageCoordX > 0)
				{
					CardUVRect.X -= CardBorderOffset.X;
				}
				if (LocalPageCoordY > 0)
				{
					CardUVRect.Y -= CardBorderOffset.Y;
				}
				if (LocalPageCoordX < MipMapDesc.SizeInPages.X - 1)
				{
					CardUVRect.Z += CardBorderOffset.X;
				}
				if (LocalPageCoordY < MipMapDesc.SizeInPages.Y - 1)
				{
					CardUVRect.W += CardBorderOffset.Y;
				}
			}

			PageTableEntry.CardUVRect = CardUVRect;

			PageTableIndicesToUpdateInBuffer.Add(PageTableIndex);
		}

		Card.UpdateMinMaxAllocatedLevel();
		CardIndicesToUpdateInBuffer.Add(CardIndex);
	}

	if (bAllowCardSharing && bLockPages && Card.CardSharingId.IsValid())
	{
		check(ResLevel == Card.MinAllocatedResLevel && Card.CardSharingListIndex == INDEX_NONE);
		PendingAddCardSharingInfos.Add(FLumenCardSharingInfoPendingAdd(Card.CardSharingId, CardIndex, Card.MinAllocatedResLevel, Card.bAxisXFlipped));
	}
}

void FLumenSceneData::FreeVirtualSurface(FLumenCard& Card, uint8 FromResLevel, uint8 ToResLevel)
{
	if (Card.IsAllocated())
	{
		for (uint8 ResLevel = FromResLevel; ResLevel <= ToResLevel; ++ResLevel)
		{
			FLumenSurfaceMipMap& MipMap = Card.GetMipMap(ResLevel);

			if (MipMap.IsAllocated())
			{
				if (bAllowCardSharing && ResLevel == Card.MinAllocatedResLevel && Card.CardSharingId.IsValid() && Card.CardSharingListIndex >= 0)
				{
					check(MipMap.bLocked);
					PendingRemoveCardSharingInfos.Add(FLumenCardSharingInfoPendingRemove(Card.CardSharingId, Card.CardSharingListIndex));
					Card.CardSharingListIndex = INDEX_NONE;
				}

				// Unmap pages
				for (int32 LocalPageIndex = 0; LocalPageIndex < int32(MipMap.SizeInPagesX * MipMap.SizeInPagesY); ++LocalPageIndex)
				{
					const int32 PageTableIndex = MipMap.GetPageTableIndex(LocalPageIndex);

					FLumenPageTableEntry& PageTableEntry = PageTable[PageTableIndex];
					UnmapSurfaceCachePage(MipMap.bLocked, PageTableEntry, PageTableIndex);
					PageTableEntry = FLumenPageTableEntry();
				}

				if (MipMap.PageTableSpanSize > 0)
				{
					PageTable.RemoveSpan(MipMap.PageTableSpanOffset, MipMap.PageTableSpanSize);

					for (int32 SpanOffset = 0; SpanOffset < int32(MipMap.PageTableSpanSize); ++SpanOffset)
					{
						PageTableIndicesToUpdateInBuffer.Add(MipMap.PageTableSpanOffset + SpanOffset);
					}

					MipMap.PageTableSpanOffset = -1;
					MipMap.PageTableSpanSize = 0;
					MipMap.bLocked = false;
				}
			}
		}

		Card.UpdateMinMaxAllocatedLevel();
	}
}

/**
 * Remove any empty virtual mip allocations, and flatten page search by walking 
 * though the sparse mip maps and reusing lower res resident pages
 */
void FLumenSceneData::UpdateCardMipMapHierarchy(FLumenCard& Card)
{
	// Remove any mip map virtual allocations, which don't have any pages mapped
	for (int32 ResLevel = Card.MinAllocatedResLevel; ResLevel <= Card.MaxAllocatedResLevel; ++ResLevel)
	{
		FLumenSurfaceMipMap& MipMap = Card.GetMipMap(ResLevel);

		if (MipMap.IsAllocated())
		{
			bool IsAnyPageMapped = false;

			for (int32 LocalPageIndex = 0; LocalPageIndex < int32(MipMap.SizeInPagesX * MipMap.SizeInPagesY); ++LocalPageIndex)
			{
				const int32 PageIndex = MipMap.GetPageTableIndex(LocalPageIndex);
				if (GetPageTableEntry(PageIndex).IsMapped())
				{
					IsAnyPageMapped = true;
					break;
				}
			}

			if (!IsAnyPageMapped)
			{
				FreeVirtualSurface(Card, ResLevel, ResLevel);
			}
		}
	}
	Card.UpdateMinMaxAllocatedLevel();


	int32 ParentResLevel = Card.MinAllocatedResLevel;

	for (int32 ResLevel = ParentResLevel + 1; ResLevel <= Card.MaxAllocatedResLevel; ++ResLevel)
	{
		FLumenSurfaceMipMap& MipMap = Card.GetMipMap(ResLevel);

		if (MipMap.PageTableSpanSize > 0)
		{
			for (int32 LocalPageIndex = 0; LocalPageIndex < int32(MipMap.SizeInPagesX * MipMap.SizeInPagesY); ++LocalPageIndex)
			{
				const int32 PageIndex = MipMap.GetPageTableIndex(LocalPageIndex);
				FLumenPageTableEntry& PageTableEntry = GetPageTableEntry(PageIndex);

				if (!PageTableEntry.IsMapped())
				{
					FIntPoint LocalPageCoord;
					LocalPageCoord.X = LocalPageIndex % MipMap.SizeInPagesX;
					LocalPageCoord.Y = LocalPageIndex / MipMap.SizeInPagesX;

					FLumenSurfaceMipMap& ParentMipMap = Card.GetMipMap(ParentResLevel);
					const FIntPoint ParentLocalPageCoord = (LocalPageCoord * ParentMipMap.GetSizeInPages()) / MipMap.GetSizeInPages();
					const int32 ParentLocalPageIndex = ParentLocalPageCoord.X + ParentLocalPageCoord.Y * ParentMipMap.SizeInPagesX;

					const int32 ParentPageIndex = ParentMipMap.GetPageTableIndex(ParentLocalPageIndex);
					FLumenPageTableEntry& ParentPageTableEntry = GetPageTableEntry(ParentPageIndex);

					PageTableEntry.SamplePageIndex = ParentPageTableEntry.SamplePageIndex;
					PageTableEntry.SampleAtlasBiasX = ParentPageTableEntry.SampleAtlasBiasX;
					PageTableEntry.SampleAtlasBiasY = ParentPageTableEntry.SampleAtlasBiasY;
					PageTableEntry.SampleCardResLevelX = ParentPageTableEntry.SampleCardResLevelX;
					PageTableEntry.SampleCardResLevelY = ParentPageTableEntry.SampleCardResLevelY;

					PageTableIndicesToUpdateInBuffer.Add(PageIndex);
				}
			}

			ParentResLevel = ResLevel;
		}
	}
}

void FLumenSceneData::CopyInitialData(const FLumenSceneData& SourceSceneData)
{
	// bTrackAllPrimitives is assumed to be the same across all FLumenSceneData
	bTrackAllPrimitives = SourceSceneData.bTrackAllPrimitives;

	PrimitiveGroups = SourceSceneData.PrimitiveGroups;
	PrimitiveCullingInfos = SourceSceneData.PrimitiveCullingInfos;
	InstanceCullingInfos = SourceSceneData.InstanceCullingInfos;
	RayTracingGroups = SourceSceneData.RayTracingGroups;
	LandscapePrimitives = SourceSceneData.LandscapePrimitives;
	PendingAddOperations = SourceSceneData.PendingAddOperations;
	PendingUpdateOperations = SourceSceneData.PendingUpdateOperations;
	PendingRemoveOperations = SourceSceneData.PendingRemoveOperations;

	// Newly created scene data has no mesh cards yet, so clear mesh card indices
	for (FLumenPrimitiveGroup& PrimitiveGroup : PrimitiveGroups)
	{
		PrimitiveGroup.MeshCardsIndex = -1;
	}

	for (FLumenPrimitiveGroupCullingInfo& CullingInfo : PrimitiveCullingInfos)
	{
		CullingInfo.bVisible = false;
	}

	for (FLumenPrimitiveGroupCullingInfo& CullingInfo : InstanceCullingInfos)
	{
		CullingInfo.bVisible = false;
	}
}

#if WITH_MGPU
// When the GPU mask changes for a view specific Lumen scene, we copy all data cross GPU to avoid transient glitches and
// potentially corrupt data.  Copying is expensive, but the assumption is that the GPU index will only change for debugging
// or performance tweaks, not during steady rendering.
void FLumenSceneData::UpdateGPUMask(FRDGBuilder& GraphBuilder, const FLumenSceneFrameTemporaries& FrameTemporaries, FLumenViewState& LumenViewState, FRHIGPUMask ViewGPUMask)
{
	check(bViewSpecific == true);

	if (!bViewSpecificMaskInitialized)
	{
		ViewSpecificMask = ViewGPUMask;
		bViewSpecificMaskInitialized = true;
	}
	else
	{
		if (ViewSpecificMask != ViewGPUMask)
		{
			FLumenSceneData* LumenSceneData = this;
			FRHIGPUMask SourceGPUMask = ViewSpecificMask;
			FRHIGPUMask DestGPUMask = ViewGPUMask;

			ViewSpecificMask = ViewGPUMask;

			// Since we're copying all GPU state cross-GPU, it should now be coherent, and we can copy the per-GPU page state as well
			for (uint32 DestGPUIndex : FRHIGPUMask::All())
			{
				if (!SourceGPUMask.Contains(DestGPUIndex))
				{
					// Copy operator is deleted, need to manually copy item by item
					const FBinaryHeap<uint32, uint32>& SourceHeap = LastCapturedPageHeap[SourceGPUMask.GetFirstIndex()];
					FBinaryHeap<uint32, uint32>& DestHeap = LastCapturedPageHeap[DestGPUIndex];

					uint32 IndexSize = SourceHeap.GetIndexSize();
					for (uint32 Index = 0; Index < IndexSize; Index++)
					{
						// We add items to heaps in tandem, so all should have the same indices
						check(SourceHeap.IsPresent(Index) == DestHeap.IsPresent(Index));

						if (SourceHeap.IsPresent(Index))
						{
							DestHeap.Update(SourceHeap.GetKey(Index), Index);
						}
					}
				}
			}

			// Make array of resources from FLumenSceneFrameTemporaries, since it's on the stack, and can't be passed to the Pass
			TArray<FRDGTextureRef, TFixedAllocator<10>> FrameTemporaryTextures;

			#define ADD_LUMEN_FRAME_TEMPORARY(NAME) \
				if (FrameTemporaries.NAME) FrameTemporaryTextures.Add(FrameTemporaries.NAME)

			ADD_LUMEN_FRAME_TEMPORARY(AlbedoAtlas);
			ADD_LUMEN_FRAME_TEMPORARY(NormalAtlas);
			ADD_LUMEN_FRAME_TEMPORARY(EmissiveAtlas);
			ADD_LUMEN_FRAME_TEMPORARY(DepthAtlas);

			ADD_LUMEN_FRAME_TEMPORARY(DirectLightingAtlas);
			ADD_LUMEN_FRAME_TEMPORARY(IndirectLightingAtlas);
			ADD_LUMEN_FRAME_TEMPORARY(RadiosityNumFramesAccumulatedAtlas);
			ADD_LUMEN_FRAME_TEMPORARY(FinalLightingAtlas);

			ADD_LUMEN_FRAME_TEMPORARY(DiffuseLightingAndSecondMomentHistoryAtlas);
			ADD_LUMEN_FRAME_TEMPORARY(NumFramesAccumulatedHistoryAtlas);

			#undef ADD_LUMEN_FRAME_TEMPORARY

			AddPass(GraphBuilder, RDG_EVENT_NAME("LumenCrossGPUTransfer"),
				[LumenSceneData, LumenView = &LumenViewState, FrameTemporaryTextures, SourceGPUMask, DestGPUMask](FRHICommandList& RHICmdList)
			{
				uint32 SourceGPUIndex = SourceGPUMask.GetFirstIndex();

				TArray<FTransferResourceParams> CrossGPUTransferResources;
				for (uint32 DestGPUIndex : FRHIGPUMask::All())
				{
					if (!SourceGPUMask.Contains(DestGPUIndex))
					{
						#define TRANSFER_LUMEN_RESOURCE(NAME) \
							if (LumenSceneData->NAME) CrossGPUTransferResources.Add(FTransferResourceParams(LumenSceneData->NAME->GetRHI(), SourceGPUIndex, DestGPUIndex, false, false))

						TRANSFER_LUMEN_RESOURCE(CardBuffer);
						TRANSFER_LUMEN_RESOURCE(MeshCardsBuffer);
						TRANSFER_LUMEN_RESOURCE(HeightfieldBuffer);
						TRANSFER_LUMEN_RESOURCE(SceneInstanceIndexToMeshCardsIndexBuffer);
						TRANSFER_LUMEN_RESOURCE(CardPageBuffer);

						TRANSFER_LUMEN_RESOURCE(CardPageLastUsedBuffer);
						TRANSFER_LUMEN_RESOURCE(CardPageHighResLastUsedBuffer);

						TRANSFER_LUMEN_RESOURCE(AlbedoAtlas);
						TRANSFER_LUMEN_RESOURCE(NormalAtlas);
						TRANSFER_LUMEN_RESOURCE(EmissiveAtlas);
						TRANSFER_LUMEN_RESOURCE(DepthAtlas);

						TRANSFER_LUMEN_RESOURCE(DirectLightingAtlas);
						TRANSFER_LUMEN_RESOURCE(IndirectLightingAtlas);
						TRANSFER_LUMEN_RESOURCE(RadiosityNumFramesAccumulatedAtlas);
						TRANSFER_LUMEN_RESOURCE(FinalLightingAtlas);
						TRANSFER_LUMEN_RESOURCE(TileShadowDownsampleFactorAtlas);

						TRANSFER_LUMEN_RESOURCE(DiffuseLightingAndSecondMomentHistoryAtlas);
						TRANSFER_LUMEN_RESOURCE(NumFramesAccumulatedHistoryAtlas);

						TRANSFER_LUMEN_RESOURCE(RadiosityTraceRadianceAtlas);
						TRANSFER_LUMEN_RESOURCE(RadiosityTraceHitDistanceAtlas);
						TRANSFER_LUMEN_RESOURCE(RadiosityProbeSHRedAtlas);
						TRANSFER_LUMEN_RESOURCE(RadiosityProbeSHGreenAtlas);
						TRANSFER_LUMEN_RESOURCE(RadiosityProbeSHBlueAtlas);

						TRANSFER_LUMEN_RESOURCE(PageTableBuffer);

						#undef TRANSFER_LUMEN_RESOURCE

						for (FRDGTextureRef Texture : FrameTemporaryTextures)
						{
							CrossGPUTransferResources.Add(FTransferResourceParams(Texture->GetRHI(), SourceGPUIndex, DestGPUIndex, false, false));
						}

						// Also transfer the view state resources cross GPU
						LumenView->AddCrossGPUTransfers(SourceGPUIndex, DestGPUIndex, CrossGPUTransferResources);
					}
				}

				RHICmdList.TransferResources(CrossGPUTransferResources);
			});
		}
	}
}
#endif  // WITH_MGPU

/**
 * Evict all pages on demand, useful for debugging
 */
void FLumenSceneData::ForceEvictEntireCache()
{
	TSparseUniqueList<int32, SceneRenderingAllocator> DirtyCards;

	while (EvictOldestAllocation(/*MaxFramesSinceLastUsed*/ 0, DirtyCards))
	{
	}

	for (int32 CardIndex : DirtyCards.Array)
	{
		FLumenCard& Card = Cards[CardIndex];
		UpdateCardMipMapHierarchy(Card);
		CardIndicesToUpdateInBuffer.Add(CardIndex);
	}
}

bool FLumenSceneData::EvictOldestAllocation(uint32 MaxFramesSinceLastUsed, TSparseUniqueList<int32, SceneRenderingAllocator>& DirtyCards)
{
	if (UnlockedAllocationHeap.Num() > 0)
	{
		const uint32 PageTableIndex = UnlockedAllocationHeap.Top();
		const uint32 LastFrameUsed = UnlockedAllocationHeap.GetKey(PageTableIndex);

		if (uint32(LastFrameUsed + MaxFramesSinceLastUsed) <= SurfaceCacheFeedback.GetFrameIndex())
		{
			UnlockedAllocationHeap.Pop();

			FLumenPageTableEntry& Page = PageTable[PageTableIndex];
			if (Page.IsMapped())
			{
				UnmapSurfaceCachePage(false, Page, PageTableIndex);
				DirtyCards.Add(Page.CardIndex);
			}

			return true;
		}
	}

	return false;
}

void FLumenSceneData::DumpStats(const FDistanceFieldSceneData& DistanceFieldSceneData, bool bDumpMeshDistanceFields, bool bDumpPrimitiveCullingInfos, bool bDumpPrimitiveGroups)
{
	const FIntPoint PageAtlasSizeInPages = GetDesiredPhysicalAtlasSizeInPages(SurfaceCacheResolution);
	const int32 NumPhysicalPages = PageAtlasSizeInPages.X * PageAtlasSizeInPages.Y;

	int32 NumCards = 0;
	int32 NumVisibleCards = 0;
	FLumenCard::FSurfaceStats SurfaceStats;

	for (int32 CardIndex = 0; CardIndex < CardCullingInfos.Num(); ++CardIndex)
	{
		if (CardCullingInfos.IsAllocated(CardIndex))
		{
			const FLumenCard& Card = Cards[CardIndex];
			const FLumenCardCullingInfo& CardCullingInfo = CardCullingInfos[CardIndex];

			++NumCards;

			if (CardCullingInfo.bVisible)
			{
				++NumVisibleCards;

				Card.GetSurfaceStats(CardCullingInfo, PageTable, SurfaceStats);
			}
		}
	}

	int32 NumPrimitiveGroups = 0;
	int32 NumPrimitivesMerged = 0;
	int32 NumInstancesMerged = 0;
	int32 NumMeshCards = 0;
	uint32 NumFarFieldPrimitiveGroups = 0;
	uint32 NumFarFieldMeshCards = 0;
	uint32 NumFarFieldCards = 0;
	FLumenCard::FSurfaceStats FarFieldSurfaceStats;
	SIZE_T PrimitiveGroupsAllocatedMemory = PrimitiveGroups.GetAllocatedSize();

	PrimitiveGroupsAllocatedMemory += PrimitiveCullingInfos.GetAllocatedSize();
	PrimitiveGroupsAllocatedMemory += InstanceCullingInfos.GetAllocatedSize();

	for (const FLumenPrimitiveGroup& PrimitiveGroup : PrimitiveGroups)
	{
		++NumPrimitiveGroups;

		if (PrimitiveGroup.HasMergedInstances())
		{
			for (const FPrimitiveSceneInfo* ScenePrimitive : PrimitiveGroup.GetPrimitives())
			{
				++NumPrimitivesMerged;
				NumInstancesMerged += ScenePrimitive->GetNumInstanceSceneDataEntries();
			}
		}

		if (PrimitiveGroup.MeshCardsIndex >= 0)
		{
			++NumMeshCards;
		}

		if (PrimitiveGroup.IsFarField())
		{
			++NumFarFieldPrimitiveGroups;

			if (PrimitiveGroup.MeshCardsIndex >= 0)
			{
				++NumFarFieldMeshCards;

				const FLumenMeshCards& MeshCardsInstance = MeshCards[PrimitiveGroup.MeshCardsIndex];
				NumFarFieldCards += MeshCardsInstance.NumCards;

				for (uint32 LocalCardIndex = 0; LocalCardIndex < MeshCardsInstance.NumCards; ++LocalCardIndex)
				{
					const FLumenCard& LumenCard = Cards[MeshCardsInstance.FirstCardIndex + LocalCardIndex];
					const FLumenCardCullingInfo& LumenCardCullingInfo = CardCullingInfos[MeshCardsInstance.FirstCardIndex + LocalCardIndex];
					if (LumenCard.IsAllocated())
					{
						LumenCard.GetSurfaceStats(LumenCardCullingInfo, PageTable, FarFieldSurfaceStats);
					}
				}
			}
		}

		PrimitiveGroupsAllocatedMemory += PrimitiveGroup.GetPrimitivesArrayAllocatedSize();
	}

	FLumenSurfaceCacheAllocator::FStats AllocatorStats;
	SurfaceCacheAllocator.GetStats(AllocatorStats);

	UE_LOGF(LogRenderer, Log, "*** LumenScene Stats ***");
	UE_LOGF(LogRenderer, Log, "  Mesh SDF Objects: %d", DistanceFieldSceneData.NumObjectsInBuffer);
	UE_LOGF(LogRenderer, Log, "  Primitive groups: %d", NumPrimitiveGroups);
	UE_LOGF(LogRenderer, Log, "  Merged primitives: %d", NumPrimitivesMerged);
	UE_LOGF(LogRenderer, Log, "  Merged instances: %d", NumInstancesMerged);
	UE_LOGF(LogRenderer, Log, "  Mesh cards: %d", NumMeshCards);
	UE_LOGF(LogRenderer, Log, "  Cards: %d", NumCards);

	UE_LOGF(LogRenderer, Log, "*** Surface cache ***");
	UE_LOGF(LogRenderer, Log, "  Allocated %d physical pages out of %d", NumPhysicalPages - AllocatorStats.NumFreePages, NumPhysicalPages);
	UE_LOGF(LogRenderer, Log, "  Bin pages: %d, wasted pages: %d, free texels: %.3fM", AllocatorStats.BinNumPages, AllocatorStats.BinNumWastedPages, AllocatorStats.BinPageFreeTexels / (1024.0f * 1024.0f));
	UE_LOGF(LogRenderer, Log, "  Virtual texels: %.3fM", SurfaceStats.NumVirtualTexels / (1024.0f * 1024.0f));
	UE_LOGF(LogRenderer, Log, "  Locked virtual texels: %.3fM", SurfaceStats.NumLockedVirtualTexels / (1024.0f * 1024.0f));
	UE_LOGF(LogRenderer, Log, "  Physical texels: %.3fM, usage: %.3f%%", SurfaceStats.NumPhysicalTexels / (1024.0f * 1024.0f), (100.0f * SurfaceStats.NumPhysicalTexels) / (PhysicalAtlasSize.X * PhysicalAtlasSize.Y));
	UE_LOGF(LogRenderer, Log, "  Locked Physical texels: %.3fM, usage: %.3f%%", SurfaceStats.NumLockedPhysicalTexels / (1024.0f * 1024.0f), (100.0f * SurfaceStats.NumLockedPhysicalTexels) / (PhysicalAtlasSize.X * PhysicalAtlasSize.Y));
	UE_LOGF(LogRenderer, Log, "  Dropped res levels: %u", SurfaceStats.DroppedResLevels);
	UE_LOGF(LogRenderer, Log, "  Mesh cards to add: %d", NumMeshCardsToAdd);
	UE_LOGF(LogRenderer, Log, "  Locked cards to update: %d", NumLockedCardsToUpdate);
	UE_LOGF(LogRenderer, Log, "  Hi-res pages to add: %d", NumHiResPagesToAdd);

	if (NumFarFieldCards > 0 || NumFarFieldMeshCards > 0)
	{
		UE_LOGF(LogRenderer, Log, "*** Far Field ***");
		UE_LOGF(LogRenderer, Log, "  Primitive groups: %d", NumFarFieldPrimitiveGroups);
		UE_LOGF(LogRenderer, Log, "  Mesh cards: %d", NumFarFieldMeshCards);
		UE_LOGF(LogRenderer, Log, "  Cards: %d", NumFarFieldCards);
		UE_LOGF(LogRenderer, Log, "  Virtual texels: %.3fM", FarFieldSurfaceStats.NumVirtualTexels / (1024.0f * 1024.0f));
		UE_LOGF(LogRenderer, Log, "  Locked virtual texels: %.3fM", FarFieldSurfaceStats.NumLockedVirtualTexels / (1024.0f * 1024.0f));
		UE_LOGF(LogRenderer, Log, "  Physical texels: %.3fM, usage: %.3f%%", FarFieldSurfaceStats.NumPhysicalTexels / (1024.0f * 1024.0f), (100.0f * FarFieldSurfaceStats.NumPhysicalTexels) / (PhysicalAtlasSize.X * PhysicalAtlasSize.Y));
		UE_LOGF(LogRenderer, Log, "  Locked Physical texels: %.3fM, usage: %.3f%%", FarFieldSurfaceStats.NumLockedPhysicalTexels / (1024.0f * 1024.0f), (100.0f * FarFieldSurfaceStats.NumLockedPhysicalTexels) / (PhysicalAtlasSize.X * PhysicalAtlasSize.Y));
		UE_LOGF(LogRenderer, Log, "  Dropped res levels: %u", FarFieldSurfaceStats.DroppedResLevels);
	}

	UE_LOGF(LogRenderer, Log, "*** Surface cache Bin Allocator ***");
	for (const FLumenSurfaceCacheAllocator::FBinStats& Bin : AllocatorStats.Bins)
	{
		UE_LOGF(LogRenderer, Log, "  %3d,%3d bin has %5d allocations using %3d pages", Bin.ElementSize.X, Bin.ElementSize.Y, Bin.NumAllocations, Bin.NumPages);
	}

	UE_LOGF(LogRenderer, Log, "*** CPU Memory ***");
	UE_LOGF(LogRenderer, Log, "  Primitive groups allocated memory: %.3fMb", PrimitiveGroupsAllocatedMemory / (1024.0f * 1024.0f));
	UE_LOGF(LogRenderer, Log, "  Cards allocated memory: %.3fMb", (Cards.GetAllocatedSize() + CardCullingInfos.GetAllocatedSize()) / (1024.0f * 1024.0f));
	UE_LOGF(LogRenderer, Log, "  MeshCards allocated memory: %.3fMb", MeshCards.GetAllocatedSize() / (1024.0f * 1024.0f));

	UE_LOGF(LogRenderer, Log, "*** GPU Memory ***");
	UE_LOGF(LogRenderer, Log, "  CardBuffer: %.3fMb", TryGetSize(CardBuffer) / (1024.0f * 1024.0f));
	UE_LOGF(LogRenderer, Log, "  MeshCardsBuffer: %.3fMb", TryGetSize(MeshCardsBuffer) / (1024.0f * 1024.0f));
	UE_LOGF(LogRenderer, Log, "  PageTable: %.3fMb", TryGetSize(PageTableBuffer) / (1024.0f * 1024.0f));
	UE_LOGF(LogRenderer, Log, "  CardPages: %.3fMb", TryGetSize(CardPageBuffer) / (1024.0f * 1024.0f));
	UE_LOGF(LogRenderer, Log, "  SceneInstanceIndexToMeshCardsIndexBuffer: %.3fMb", TryGetSize(SceneInstanceIndexToMeshCardsIndexBuffer) / (1024.0f * 1024.0f));

	if (bDumpMeshDistanceFields)
	{
		DistanceFieldSceneData.ListMeshDistanceFields(true);
	}

	if (bDumpPrimitiveCullingInfos)
	{
#if STATS
		UE_LOGF(LogRenderer, Log, "*** LumenScene PrimitiveCullingInfos ***");

		for (const FLumenPrimitiveGroupCullingInfo& CullingInfo : PrimitiveCullingInfos)
		{
			if (CullingInfo.PrimitiveGroupIndex > 0)
			{
				if (PrimitiveGroups.IsAllocated(CullingInfo.PrimitiveGroupIndex))
				{
					const FLumenPrimitiveGroup& PrimitiveGroup = PrimitiveGroups[CullingInfo.PrimitiveGroupIndex];
					if (PrimitiveGroup.GetPrimitives().Num() > 0)
					{
						const FPrimitiveSceneInfo* ScenePrimitive = PrimitiveGroup.GetPrimitives()[0];
						if (ScenePrimitive && ScenePrimitive->Proxy)
						{
							UE_LOGF(LogRenderer, Log, "NumInstances:%d FarField:%d Heightfield:%d %ls",
								CullingInfo.NumInstances,
								PrimitiveGroup.IsFarField() ? 1 : 0,
								PrimitiveGroup.IsHeightfield() ? 1 : 0,
								*ScenePrimitive->Proxy->GetStatId().GetName().ToString());
						}
					}
				}
			}
		}
#endif
	}

	if (bDumpPrimitiveGroups)
	{
#if STATS
		UE_LOGF(LogRenderer, Log, "*** LumenScene Primitives ***");

		for (const FLumenPrimitiveGroup& PrimitiveGroup : PrimitiveGroups)
		{
			for (const FPrimitiveSceneInfo* ScenePrimitive : PrimitiveGroup.GetPrimitives())
			{
				if (ScenePrimitive && ScenePrimitive->Proxy)
				{
					UE_LOGF(LogRenderer, Log, "Group:%d InstanceIndex:%d FarField:%d Heightfield:%d %ls", 
						PrimitiveGroup.RayTracingGroupMapElementId.GetIndex(),
						PrimitiveGroup.PrimitiveInstanceIndex,
						PrimitiveGroup.IsFarField() ? 1 : 0,
						PrimitiveGroup.IsHeightfield() ? 1 : 0,
						*ScenePrimitive->Proxy->GetStatId().GetName().ToString());
				}
			}
		}
#endif
	}
}