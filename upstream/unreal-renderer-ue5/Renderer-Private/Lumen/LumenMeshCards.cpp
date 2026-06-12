// Copyright Epic Games, Inc. All Rights Reserved.

#include "LumenMeshCards.h"
#include "MeshCardRepresentation.h"
#include "ComponentRecreateRenderStateContext.h"
#include "LumenHeightfields.h"
#include "MeshCardBuild.h"
#include "InstanceDataSceneProxy.h"
#include "Lumen/LumenSceneData.h"
#include "RenderGraphBuilder.h"
#include "ScenePrivate.h"
#include "PrimitiveSceneInfo.h"

TAutoConsoleVariable<float> CVarLumenMeshCardsMinSize(
	TEXT("r.LumenScene.SurfaceCache.MeshCardsMinSize"),
	10.0f,
	TEXT("Minimum mesh cards world space size to be included in Lumen Scene."),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenMeshCardsMergeComponents = 1;
FAutoConsoleVariableRef CVarLumenMeshCardsMergeComponents(
	TEXT("r.LumenScene.SurfaceCache.MeshCardsMergeComponents"),
	GLumenMeshCardsMergeComponents,
	TEXT("Whether to merge all components with the same RayTracingGroupId into a single MeshCards."),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenMeshCardsMergeInstances = 0;
FAutoConsoleVariableRef CVarLumenMeshCardsMergeInstances(
	TEXT("r.LumenScene.SurfaceCache.MeshCardsMergeInstances"),
	GLumenMeshCardsMergeInstances,
	TEXT("Whether to merge all instances of a Instanced Static Mesh Component into a single MeshCards."),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenMeshCardsMergedCardMinSurfaceArea = 0.05f;
FAutoConsoleVariableRef CVarLumenMeshCardsMergedCardMinSurfaceArea(
	TEXT("r.LumenScene.SurfaceCache.MeshCardsMergedCardMinSurfaceArea"),
	GLumenMeshCardsMergedCardMinSurfaceArea,
	TEXT("Minimum area to spawn a merged card."),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenMeshCardsMergeInstancesMaxSurfaceAreaRatio = 1.7f;
FAutoConsoleVariableRef CVarLumenMeshCardsMergeInstancesMaxSurfaceAreaRatio(
	TEXT("r.LumenScene.SurfaceCache.MeshCardsMergeInstancesMaxSurfaceAreaRatio"),
	GLumenMeshCardsMergeInstancesMaxSurfaceAreaRatio,
	TEXT("Only merge if the (combined box surface area) / (summed instance box surface area) < MaxSurfaceAreaRatio"),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenMeshCardsMergedResolutionScale = .3f;
FAutoConsoleVariableRef CVarLumenMeshCardsMergedResolutionScale(
	TEXT("r.LumenScene.SurfaceCache.MeshCardsMergedResolutionScale"),
	GLumenMeshCardsMergedResolutionScale,
	TEXT("Scale on the resolution calculation for a merged MeshCards.  This compensates for the merged box getting a higher resolution assigned due to being closer to the viewer."),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenMeshCardsMergedMaxWorldSize = 10000.0f;
FAutoConsoleVariableRef CVarLumenMeshCardsMergedMaxWorldSize(
	TEXT("r.LumenScene.SurfaceCache.MeshCardsMergedMaxWorldSize"),
	GLumenMeshCardsMergedMaxWorldSize,
	TEXT("Only merged bounds less than this size on any axis are considered, since Lumen Scene streaming relies on object granularity."),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenMeshCardsCullFaces = 1;
FAutoConsoleVariableRef CVarLumenMeshCardsCullFaces(
	TEXT("r.LumenScene.SurfaceCache.MeshCardsCullFaces"),
	GLumenMeshCardsCullFaces,
	TEXT(""),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenMeshCardsDebugSingleCard = -1;
FAutoConsoleVariableRef CVarLumenMeshCardsDebugSingleCard(
	TEXT("r.LumenScene.SurfaceCache.MeshCardsDebugSingleCard"),
	GLumenMeshCardsDebugSingleCard,
	TEXT("Spawn only a specified card on mesh. Useful for debugging."),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarLumenSurfaceCacheHeightfieldCaptureMargin(
	TEXT("r.Lumen.SurfaceCache.HeightfieldCaptureMargin"),
	100.0f,
	TEXT("Amount to expand heightfield component bbox for card capture purposes."),
	ECVF_RenderThreadSafe
);

float LumenMeshCards::GetCardMinSurfaceArea(bool bEmissiveLightSource)
{
	const float MeshCardsMinSize = CVarLumenMeshCardsMinSize.GetValueOnRenderThread();
	return MeshCardsMinSize * MeshCardsMinSize * (bEmissiveLightSource ? 0.2f : 1.0f);
}

class FLumenCardGPUData
{
public:
	// Must match usf
	enum { DataStrideInFloat4s = 7 };
	enum { DataStrideInBytes = DataStrideInFloat4s * sizeof(FVector4f) };

	static void PackSurfaceMipMap(const FLumenCard& Card, int32 ResLevel, uint32& PackedSizeInPages, uint32& PackedPageTableOffset)
	{
		PackedSizeInPages = 0;
		PackedPageTableOffset = 0;

		if (Card.IsAllocated())
		{
			const FLumenSurfaceMipMap& MipMap = Card.GetMipMap(ResLevel);

			if (MipMap.IsAllocated())
			{
				PackedSizeInPages = MipMap.SizeInPagesX | (MipMap.SizeInPagesY << 16);
				PackedPageTableOffset = MipMap.PageTableSpanOffset;
			}
		}
	}

	static void FillData(bool bCardVisible, const FLumenCard& RESTRICT Card, const FLumenPrimitiveGroup* InPrimitiveGroup, FVector4f* RESTRICT OutData)
	{
		// Note: layout must match GetLumenCardData in usf

		const FLumenCardOBBd& WorldOBB = Card.GetWorldOBB();
		const FDFVector3 WorldPosition(WorldOBB.Origin);

		const FIntPoint ResLevelBias = Card.ResLevelToResLevelXYBias();
		const uint32 LightingChannelMask = InPrimitiveGroup ? InPrimitiveGroup->GetLightingChannelMask() : 0xF;

		uint32 Packed0W = 0;
		Packed0W = uint8(ResLevelBias.X) & 0xFF;
		Packed0W |= (uint8(ResLevelBias.Y) & 0xFF) << 8;
		Packed0W |= (uint8(Card.AxisAlignedDirectionIndex) & 0xF) << 16;
		Packed0W |= (LightingChannelMask & 0xF) << 20;
		Packed0W |= bCardVisible && Card.IsAllocated() ? (1 << 24) : 0;
		Packed0W |= Card.bHeightfield && Card.IsAllocated() ? (1 << 25) : 0;
		Packed0W |= Card.bFarField && Card.IsAllocated() ? (1 << 26) : 0;

		OutData[0] = FVector4f(WorldPosition.High, *(float*)&Packed0W);
		OutData[1] = FVector4f(WorldOBB.AxisX[0], WorldOBB.AxisY[0], WorldOBB.AxisZ[0], WorldPosition.Low.X);
		OutData[2] = FVector4f(WorldOBB.AxisX[1], WorldOBB.AxisY[1], WorldOBB.AxisZ[1], WorldPosition.Low.Y);
		OutData[3] = FVector4f(WorldOBB.AxisX[2], WorldOBB.AxisY[2], WorldOBB.AxisZ[2], WorldPosition.Low.Z);

		// Map low-res level for diffuse
		uint32 PackedSizeInPages = 0;
		uint32 PackedPageTableOffset = 0;
		PackSurfaceMipMap(Card, Card.MinAllocatedResLevel, PackedSizeInPages, PackedPageTableOffset);

		// Map hi-res for specular
		uint32 PackedHiResSizeInPages = 0;
		uint32 PackedHiResPageTableOffset = 0;
		PackSurfaceMipMap(Card, Card.MaxAllocatedResLevel, PackedHiResSizeInPages, PackedHiResPageTableOffset);

		OutData[4].X = *((float*)&PackedSizeInPages);
		OutData[4].Y = *((float*)&PackedPageTableOffset);
		OutData[4].Z = *((float*)&PackedHiResSizeInPages);
		OutData[4].W = *((float*)&PackedHiResPageTableOffset);

		float AverageTexelSize = 100.0f;
		if (Card.IsAllocated())
		{
			FLumenMipMapDesc MipMapDesc;
			Card.GetMipMapDesc(Card.MinAllocatedResLevel, MipMapDesc);
			AverageTexelSize = 0.5f * (WorldOBB.Extent.X / MipMapDesc.Resolution.X + WorldOBB.Extent.Y / MipMapDesc.Resolution.Y);
		}

		const FLumenCardOBBf MeshCardsOBB = Card.GetMeshCardsOBB();
		// Transpose the axes so they rotate a mesh cards space vector to card space
		const int32 MeshCardsAxisX = MeshCardRepresentation::GetAxisAlignedDirectionIndex(FVector3f(MeshCardsOBB.AxisX[0], MeshCardsOBB.AxisY[0], MeshCardsOBB.AxisZ[0]));
		const int32 MeshCardsAxisY = MeshCardRepresentation::GetAxisAlignedDirectionIndex(FVector3f(MeshCardsOBB.AxisX[1], MeshCardsOBB.AxisY[1], MeshCardsOBB.AxisZ[1]));
		const int32 MeshCardsAxisZ = MeshCardRepresentation::GetAxisAlignedDirectionIndex(FVector3f(MeshCardsOBB.AxisX[2], MeshCardsOBB.AxisY[2], MeshCardsOBB.AxisZ[2]));
		check(MeshCardsAxisX >= 0 && MeshCardsAxisX <= 5 && MeshCardsAxisY >= 0 && MeshCardsAxisY <= 5 && MeshCardsAxisZ >= 0 && MeshCardsAxisZ <= 5);
		check(MeshCardsAxisX / 2 != MeshCardsAxisY / 2 && MeshCardsAxisY / 2 != MeshCardsAxisZ / 2 && MeshCardsAxisZ / 2 != MeshCardsAxisX / 2);
		const uint32 PackedMeshCardsAxes = (MeshCardsAxisZ << 6u) | (MeshCardsAxisY << 3u) | MeshCardsAxisX;

		OutData[5] = FVector4f(MeshCardsOBB.Origin, *(float*)&PackedMeshCardsAxes);
		OutData[6] = FVector4f(MeshCardsOBB.Extent, AverageTexelSize);

		static_assert(DataStrideInFloat4s == 7, "Data stride doesn't match");
	}
};

struct FLumenMeshCardsGPUData
{
	// Must match LUMEN_MESH_CARDS_DATA_STRIDE in LumenCardCommon.ush
	enum { DataStrideInFloat4s = 6 };
	enum { DataStrideInBytes = DataStrideInFloat4s * 16 };

	static void FillData(const class FLumenMeshCards& RESTRICT MeshCards, FVector4f* RESTRICT OutData);
};

void FLumenMeshCardsGPUData::FillData(const FLumenMeshCards& RESTRICT MeshCards, FVector4f* RESTRICT OutData)
{
	// Note: layout must match GetLumenMeshCardsData in usf

	const FDFVector3 WorldOrigin(MeshCards.LocalToWorld.GetOrigin());

	OutData[0] = WorldOrigin.High;
	OutData[1] = FVector4f(FVector4(MeshCards.WorldToLocalRotation.GetScaledAxis(EAxis::X), WorldOrigin.Low.X));
	OutData[2] = FVector4f(FVector4(MeshCards.WorldToLocalRotation.GetScaledAxis(EAxis::Y), WorldOrigin.Low.Y));
	OutData[3] = FVector4f(FVector4(MeshCards.WorldToLocalRotation.GetScaledAxis(EAxis::Z), WorldOrigin.Low.Z));

	uint32 PackedData[4];
	PackedData[0] = MeshCards.FirstCardIndex;
	PackedData[1] = MeshCards.NumCards & 0xFFFF;
	PackedData[1] |= MeshCards.bHeightfield ? 0x10000 : 0;
	PackedData[1] |= MeshCards.bMostlyTwoSided ? 0x20000 : 0;
	PackedData[2] = MeshCards.CardLookup[0];
	PackedData[3] = MeshCards.CardLookup[1];
	OutData[4] = *(FVector4f*)&PackedData;

	PackedData[0] = MeshCards.CardLookup[2];
	PackedData[1] = MeshCards.CardLookup[3];
	PackedData[2] = MeshCards.CardLookup[4];
	PackedData[3] = MeshCards.CardLookup[5];
	OutData[5] = *(FVector4f*)&PackedData;

	static_assert(DataStrideInFloat4s == 6, "Data stride doesn't match");
}

struct FLumenPrimitiveGroupGPUData
{
	// Must match LUMEN_PRIMITIVE_GROUP_DATA_STRIDE in LumenScene.usf
	enum { DataStrideInFloat4s = 2 };
	enum { DataStrideInBytes = DataStrideInFloat4s * 16 };

	static void FillData(const class FLumenPrimitiveGroup& RESTRICT PrimitiveGroup, const FLumenPrimitiveGroupCullingInfo* RESTRICT CullingInfo, FVector4f* RESTRICT OutData);
};

void FLumenPrimitiveGroupGPUData::FillData(const FLumenPrimitiveGroup& RESTRICT PrimitiveGroup, const FLumenPrimitiveGroupCullingInfo* RESTRICT CullingInfo, FVector4f* RESTRICT OutData)
{
	// Note: layout must match GetLumenPrimitiveGroupData in usf

	if (CullingInfo)
	{
		OutData[0] = CullingInfo->WorldSpaceBoundingBox.GetCenter();
		OutData[1] = CullingInfo->WorldSpaceBoundingBox.GetExtent();
	}
	else
	{
		OutData[0] = FVector4f::Zero();
		OutData[1] = FVector4f::Zero();
	}

	uint32 MeshCardsIndex = PrimitiveGroup.MeshCardsIndex >= 0 ? PrimitiveGroup.MeshCardsIndex : UINT32_MAX;
	OutData[0].W = *((float*) &MeshCardsIndex);

	uint32 PackedFlags = 0;
	PackedFlags |= PrimitiveGroup.HasValidMeshCards()		? 0x01 : 0;
	PackedFlags |= PrimitiveGroup.IsFarField()				? 0x02 : 0;
	PackedFlags |= PrimitiveGroup.IsHeightfield()			? 0x04 : 0;
	PackedFlags |= PrimitiveGroup.IsEmissiveLightSource()	? 0x08 : 0;
	OutData[1].W = *((float*) &PackedFlags);

	static_assert(DataStrideInFloat4s == 2, "Data stride doesn't match");
}

void UpdateLumenMeshCards(FRDGBuilder& GraphBuilder, FRDGScatterUploadBuilder& UploadBuilder, const FScene& Scene, const FDistanceFieldSceneData& DistanceFieldSceneData, FLumenSceneFrameTemporaries& FrameTemporaries, FLumenSceneData& LumenSceneData)
{
	LLM_SCOPE_BYTAG(Lumen);
	QUICK_SCOPE_CYCLE_COUNTER(UpdateLumenMeshCards);

	if (LumenSceneData.bReuploadSceneRequest)
	{
		LumenSceneData.HeightfieldIndicesToUpdateInBuffer.Reset();
		for (int32 i = 0; i < LumenSceneData.Heightfields.Num(); ++i)
		{
			LumenSceneData.HeightfieldIndicesToUpdateInBuffer.Add(i);
		}

		LumenSceneData.MeshCardsIndicesToUpdateInBuffer.Reset();
		for (int32 i = 0; i < LumenSceneData.MeshCards.Num(); ++i)
		{
			LumenSceneData.MeshCardsIndicesToUpdateInBuffer.Add(i);
		}
	}

	if (LumenScene::UseGPUDrivenUpdate() && (LumenSceneData.bReuploadSceneRequest || !LumenSceneData.PrimitiveGroupBuffer))
	{
		LumenSceneData.PrimitiveGroupIndicesToUpdateInBuffer.Reset();
		for (int32 i = 0; i < LumenSceneData.PrimitiveGroups.GetMaxSize(); ++i)
		{
			LumenSceneData.PrimitiveGroupIndicesToUpdateInBuffer.Add(i);
		}
	}

	// Upload primitive groups
	if (LumenScene::UseGPUDrivenUpdate())
	{
		QUICK_SCOPE_CYCLE_COUNTER(UpdatePrimitiveGroups);

		const uint32 NumPrimitiveGroups = LumenSceneData.PrimitiveGroups.GetMaxSize();
		const uint32 PrimitiveGroupNumFloat4s = FMath::RoundUpToPowerOfTwo(NumPrimitiveGroups * FLumenPrimitiveGroupGPUData::DataStrideInFloat4s);
		const uint32 PrimitiveGroupNumBytes = PrimitiveGroupNumFloat4s * sizeof(FVector4f);
		FRDGBuffer* PrimitiveGroupBuffer = ResizeStructuredBufferIfNeeded(GraphBuilder, LumenSceneData.PrimitiveGroupBuffer, PrimitiveGroupNumBytes, TEXT("Lumen.PrimitiveGroup"));
		FrameTemporaries.PrimitiveGroupBufferSRV = GraphBuilder.CreateSRV(PrimitiveGroupBuffer);

		const int32 NumPrimitiveGroupUploads = LumenSceneData.PrimitiveGroupIndicesToUpdateInBuffer.Num();

		if (NumPrimitiveGroupUploads > 0)
		{
			UploadBuilder.AddPass(
				GraphBuilder,
				LumenSceneData.PrimitiveGroupUploadBuffer,
				PrimitiveGroupBuffer,
				NumPrimitiveGroupUploads,
				FLumenPrimitiveGroupGPUData::DataStrideInBytes,
				TEXT("Lumen.PrimitiveGroupUpload"),
				[&LumenSceneData] (FRDGScatterUploader& Uploader)
			{
				FLumenPrimitiveGroup NullPrimitiveGroup;

				for (int32 Index : LumenSceneData.PrimitiveGroupIndicesToUpdateInBuffer)
				{
					if (Index < LumenSceneData.PrimitiveGroups.GetMaxSize())
					{
						const FLumenPrimitiveGroup* PrimitiveGroup = &NullPrimitiveGroup;
						const FLumenPrimitiveGroupCullingInfo* CullingInfo = nullptr;

						if (LumenSceneData.PrimitiveGroups.IsAllocated(Index))
						{
							PrimitiveGroup = &LumenSceneData.PrimitiveGroups[Index];
							CullingInfo = &LumenSceneData.GetPrimitiveGroupCullingInfo(*PrimitiveGroup);
						}

						FVector4f* Data = (FVector4f*)Uploader.Add_GetRef(Index);
						FLumenPrimitiveGroupGPUData::FillData(*PrimitiveGroup, CullingInfo, Data);
					}
				}
			});
		}
	}
	else
	{
		LumenSceneData.PrimitiveGroupIndicesToUpdateInBuffer.Empty(0);
		LumenSceneData.PrimitiveGroupUploadBuffer.Release();
		LumenSceneData.PrimitiveGroupBuffer.SafeRelease();
		FrameTemporaries.PrimitiveGroupBufferSRV = nullptr;
	}

	// Upload MeshCards
	{
		QUICK_SCOPE_CYCLE_COUNTER(UpdateMeshCards);

		const uint32 NumMeshCards = LumenSceneData.MeshCards.Num();
		const uint32 MeshCardsNumFloat4s = FMath::RoundUpToPowerOfTwo(NumMeshCards * FLumenMeshCardsGPUData::DataStrideInFloat4s);
		const uint32 MeshCardsNumBytes = MeshCardsNumFloat4s * sizeof(FVector4f);
		FRDGBuffer* MeshCardsBuffer = ResizeStructuredBufferIfNeeded(GraphBuilder, LumenSceneData.MeshCardsBuffer, MeshCardsNumBytes, TEXT("Lumen.MeshCards"));
		FrameTemporaries.MeshCardsBufferSRV = GraphBuilder.CreateSRV(MeshCardsBuffer);

		const int32 NumMeshCardsUploads = LumenSceneData.MeshCardsIndicesToUpdateInBuffer.Num();

		if (NumMeshCardsUploads > 0)
		{
			UploadBuilder.AddPass(
				GraphBuilder,
				LumenSceneData.MeshCardsUploadBuffer,
				MeshCardsBuffer,
				NumMeshCardsUploads,
				FLumenMeshCardsGPUData::DataStrideInBytes,
				TEXT("Lumen.MeshCardsUpload"),
				[&LumenSceneData] (FRDGScatterUploader& Uploader)
			{
				FLumenMeshCards NullMeshCards;

				for (int32 Index : LumenSceneData.MeshCardsIndicesToUpdateInBuffer)
				{
					if (Index < LumenSceneData.MeshCards.Num())
					{
						const FLumenMeshCards& MeshCards = LumenSceneData.MeshCards.IsAllocated(Index) ? LumenSceneData.MeshCards[Index] : NullMeshCards;

						FVector4f* Data = (FVector4f*)Uploader.Add_GetRef(Index);
						FLumenMeshCardsGPUData::FillData(MeshCards, Data);
					}
				}
			});
		}
	}

	// Upload Heightfields
	{
		QUICK_SCOPE_CYCLE_COUNTER(UpdateHeightfields);

		const uint32 NumHeightfields = LumenSceneData.Heightfields.Num();
		const uint32 HeightfieldsNumFloat4s = FMath::RoundUpToPowerOfTwo(NumHeightfields * FLumenHeightfieldGPUData::DataStrideInFloat4s);
		const uint32 HeightfieldsNumBytes = HeightfieldsNumFloat4s * sizeof(FVector4f);
		FRDGBuffer* HeightfieldBuffer = ResizeStructuredBufferIfNeeded(GraphBuilder, LumenSceneData.HeightfieldBuffer, HeightfieldsNumBytes, TEXT("Lumen.Heightfield"));
		FrameTemporaries.HeightfieldBufferSRV = GraphBuilder.CreateSRV(HeightfieldBuffer);

		const int32 NumHeightfieldsUploads = LumenSceneData.HeightfieldIndicesToUpdateInBuffer.Num();

		if (NumHeightfieldsUploads > 0)
		{
			UploadBuilder.AddPass(
				GraphBuilder,
				LumenSceneData.HeightfieldUploadBuffer,
				HeightfieldBuffer,
				NumHeightfieldsUploads,
				FLumenHeightfieldGPUData::DataStrideInBytes,
				TEXT("Lumen.HeightfieldUpload"),
				[&LumenSceneData] (FRDGScatterUploader& Uploader)
			{
				FLumenHeightfield NullHeightfield;

				for (int32 Index : LumenSceneData.HeightfieldIndicesToUpdateInBuffer)
				{
					if (Index < LumenSceneData.Heightfields.Num())
					{
						const FLumenHeightfield& Heightfield = LumenSceneData.Heightfields.IsAllocated(Index) ? LumenSceneData.Heightfields[Index] : NullHeightfield;

						FVector4f* Data = (FVector4f*)Uploader.Add_GetRef(Index);
						FLumenHeightfieldGPUData::FillData(Heightfield, LumenSceneData.MeshCards, Data);
					}
				}
			});
		}
	}

	// Upload SceneInstanceIndexToMeshCardsIndexBuffer
	{
		QUICK_SCOPE_CYCLE_COUNTER(UpdateSceneInstanceIndexToMeshCardsIndexBuffer);

		if (LumenSceneData.bReuploadSceneRequest)
		{
			LumenSceneData.PrimitivesToUpdateMeshCards.Reset();

			for (int32 PrimitiveIndex = 0; PrimitiveIndex < Scene.Primitives.Num(); ++PrimitiveIndex)
			{
				LumenSceneData.PrimitivesToUpdateMeshCards.Add(PrimitiveIndex);
			}
		}

		const int32 NumIndices = FMath::Max(FMath::RoundUpToPowerOfTwo(Scene.GPUScene.GetInstanceIdUpperBoundGPU()), 1024u);
		const uint32 IndexSizeInBytes = GPixelFormats[PF_R32_UINT].BlockBytes;
		const uint32 IndicesSizeInBytes = NumIndices * IndexSizeInBytes;
		FRDGBuffer* SceneInstanceIndexToMeshCardsIndexBuffer = ResizeByteAddressBufferIfNeeded(GraphBuilder, LumenSceneData.SceneInstanceIndexToMeshCardsIndexBuffer, IndicesSizeInBytes, TEXT("Lumen.SceneInstanceIndexToMeshCardsIndexBuffer"));
		FrameTemporaries.SceneInstanceIndexToMeshCardsIndexBufferSRV = GraphBuilder.CreateSRV(SceneInstanceIndexToMeshCardsIndexBuffer);

		uint32 NumIndexUploads = 0;

		for (int32 PrimitiveIndex : LumenSceneData.PrimitivesToUpdateMeshCards)
		{
			if (PrimitiveIndex < Scene.Primitives.Num())
			{
				const FPrimitiveSceneInfo* PrimitiveSceneInfo = Scene.Primitives[PrimitiveIndex];
				NumIndexUploads += PrimitiveSceneInfo->GetNumInstanceSceneDataEntries();
			}
		}

		if (NumIndexUploads > 0)
		{
			UploadBuilder.AddPass(
				GraphBuilder,
				LumenSceneData.SceneInstanceIndexToMeshCardsIndexUploadBuffer,
				SceneInstanceIndexToMeshCardsIndexBuffer,
				NumIndexUploads,
				IndexSizeInBytes,
				TEXT("Lumen.SceneInstanceIndexToMeshCardsIndexUploadBuffer"),
				[&LumenSceneData, &Scene, NumIndices] (FRDGScatterUploader& Uploader)
			{
				for (int32 PrimitiveIndex : LumenSceneData.PrimitivesToUpdateMeshCards)
				{
					if (PrimitiveIndex < Scene.Primitives.Num())
					{
						const FPrimitiveSceneInfo* PrimitiveSceneInfo = Scene.Primitives[PrimitiveIndex];
						const int32 NumInstances = PrimitiveSceneInfo->GetNumInstanceSceneDataEntries();
						const int32 InstanceDataOffset = PrimitiveSceneInfo->GetInstanceSceneDataOffset();

						for (int32 InstanceIndex = 0; InstanceIndex < NumInstances; ++InstanceIndex)
						{
							const int32 MeshCardsIndex = LumenSceneData.GetMeshCardsIndex(PrimitiveSceneInfo, InstanceIndex);

							int32 DestIndex = InstanceDataOffset + InstanceIndex;
							if (DestIndex < NumIndices)
							{
								Uploader.Add(DestIndex, &MeshCardsIndex);
							}
						}
					}
				}
			});
		}
	}
}

void Lumen::UpdateCardSceneBuffer(FRDGBuilder& GraphBuilder, FRDGScatterUploadBuilder& UploadBuilder, FLumenSceneFrameTemporaries& FrameTemporaries, const FSceneViewFamily& ViewFamily, FScene* Scene)
{
	LLM_SCOPE_BYTAG(Lumen);

	TRACE_CPUPROFILER_EVENT_SCOPE(UpdateCardSceneBuffer);
	QUICK_SCOPE_CYCLE_COUNTER(UpdateCardSceneBuffer);
	RDG_EVENT_SCOPE(GraphBuilder, "UpdateCardSceneBuffer");
	RDG_GPU_MASK_SCOPE(GraphBuilder, FRHIGPUMask::All());

	FLumenSceneData& LumenSceneData = *Scene->GetLumenSceneData(*ViewFamily.Views[0]);

	// CardBuffer
	{
		FRDGBuffer* CardBuffer = nullptr;

		{
			const int32 NumCardEntries = LumenSceneData.Cards.Num();
			const uint32 CardSceneNumFloat4s = NumCardEntries * FLumenCardGPUData::DataStrideInFloat4s;
			const uint32 CardSceneNumBytes = FMath::DivideAndRoundUp(CardSceneNumFloat4s, 16384u) * 16384 * sizeof(FVector4f);
			CardBuffer = ResizeStructuredBufferIfNeeded(GraphBuilder, LumenSceneData.CardBuffer, FMath::RoundUpToPowerOfTwo(CardSceneNumFloat4s) * sizeof(FVector4f), TEXT("Lumen.Cards"));
			FrameTemporaries.CardBufferSRV = GraphBuilder.CreateSRV(CardBuffer);
		}

		if (LumenSceneData.bReuploadSceneRequest)
		{
			LumenSceneData.CardIndicesToUpdateInBuffer.Reset();

			for (int32 i = 0; i < LumenSceneData.Cards.Num(); i++)
			{
				LumenSceneData.CardIndicesToUpdateInBuffer.Add(i);
			}
		}

		const int32 NumCardDataUploads = LumenSceneData.CardIndicesToUpdateInBuffer.Num();

		if (NumCardDataUploads > 0)
		{
			UploadBuilder.AddPass(
				GraphBuilder,
				LumenSceneData.CardUploadBuffer,
				CardBuffer,
				NumCardDataUploads,
				FLumenCardGPUData::DataStrideInBytes,
				TEXT("Lumen.CardUploadBuffer"),
				[&LumenSceneData] (FRDGScatterUploader& Uploader)
			{
				FLumenCard NullCard;

				for (int32 Index : LumenSceneData.CardIndicesToUpdateInBuffer)
				{
					if (Index < LumenSceneData.Cards.Num())
					{
						const bool bValidCard = LumenSceneData.Cards.IsAllocated(Index);
						const bool bCardVisible = bValidCard ? LumenSceneData.CardCullingInfos[Index].bVisible : false;
						const FLumenCard& Card = bValidCard ? LumenSceneData.Cards[Index] : NullCard;

						FLumenPrimitiveGroup* PrimitiveGroup = nullptr;
						if (Card.MeshCardsIndex >= 0)
						{
							const FLumenMeshCards& MeshCardsInstance = LumenSceneData.MeshCards[Card.MeshCardsIndex];
							if (MeshCardsInstance.PrimitiveGroupIndex >= 0)
							{
								PrimitiveGroup = &LumenSceneData.PrimitiveGroups[MeshCardsInstance.PrimitiveGroupIndex];
							}
						}

						FVector4f* Data = (FVector4f*)Uploader.Add_GetRef(Index);
						FLumenCardGPUData::FillData(bCardVisible, Card, PrimitiveGroup, Data);
					}
				}

				LumenSceneData.CardIndicesToUpdateInBuffer.Reset();
			});
		}
	}

	UpdateLumenMeshCards(GraphBuilder, UploadBuilder, *Scene, Scene->DistanceFieldSceneData, FrameTemporaries, LumenSceneData);
	LumenSceneData.bReuploadSceneRequest = false;
}

int32 FLumenSceneData::GetMeshCardsIndex(const FPrimitiveSceneInfo* PrimitiveSceneInfo, int32 InstanceIndex) const
{
	if (PrimitiveSceneInfo->LumenPrimitiveGroupIndices.Num() > 0)
	{
		const int32 IndexInArray = FMath::Min(InstanceIndex, PrimitiveSceneInfo->LumenPrimitiveGroupIndices.Num() - 1);
		const int32 PrimitiveGroupIndex = PrimitiveSceneInfo->LumenPrimitiveGroupIndices[IndexInArray];
		const FLumenPrimitiveGroup& PrimitiveGroup = PrimitiveGroups[PrimitiveGroupIndex];

		return PrimitiveGroup.MeshCardsIndex;
	}

	return -1;
}

void FLumenSceneData::UpdatePrimitiveGroupGPUBufferEntry(int32 PrimitiveGroupIndex)
{
	if (LumenScene::UseGPUDrivenUpdate())
	{
		PrimitiveGroupIndicesToUpdateInBuffer.Add(PrimitiveGroupIndex);
	}
}

FLumenPrimitiveGroupCullingInfo& FLumenSceneData::GetPrimitiveGroupCullingInfo(const FLumenPrimitiveGroup& PrimitiveGroup, bool bForcePrimitiveLevel)
{
	if (!bForcePrimitiveLevel && PrimitiveGroup.InstanceCullingInfoIndex >= 0)
	{
		check(PrimitiveGroup.PrimitiveInstanceIndex >= 0);
		return InstanceCullingInfos[PrimitiveGroup.InstanceCullingInfoIndex];
	}
	else
	{
		return PrimitiveCullingInfos[PrimitiveGroup.PrimitiveCullingInfoIndex];
	}
}

const FLumenPrimitiveGroupCullingInfo& FLumenSceneData::GetPrimitiveGroupCullingInfo(const FLumenPrimitiveGroup& PrimitiveGroup, bool bForcePrimitiveLevel) const
{
	return const_cast<FLumenSceneData*>(this)->GetPrimitiveGroupCullingInfo(PrimitiveGroup, bForcePrimitiveLevel);
}

void FLumenSceneData::RemovePrimitiveGroupCullingInfo(FLumenPrimitiveGroup& PrimitiveGroup)
{
	if (PrimitiveGroup.PrimitiveCullingInfoIndex >= 0)
	{
		if (PrimitiveCullingInfos.IsAllocated(PrimitiveGroup.PrimitiveCullingInfoIndex))
		{
			const FLumenPrimitiveGroupCullingInfo& PrimitiveCullingInfo = PrimitiveCullingInfos[PrimitiveGroup.PrimitiveCullingInfoIndex];

			if (PrimitiveCullingInfo.NumInstances > 0)
			{
				check(PrimitiveGroup.PrimitiveInstanceIndex >= 0);
				check(PrimitiveGroup.InstanceCullingInfoIndex == PrimitiveCullingInfo.InstanceCullingInfoOffset + PrimitiveGroup.PrimitiveInstanceIndex);
				InstanceCullingInfos.RemoveSpan(PrimitiveCullingInfo.InstanceCullingInfoOffset, PrimitiveCullingInfo.NumInstances);
			}

			PrimitiveCullingInfos.RemoveAt(PrimitiveGroup.PrimitiveCullingInfoIndex);
		}

		PrimitiveGroup.PrimitiveCullingInfoIndex = INDEX_NONE;

		if (PrimitiveGroup.InstanceCullingInfoIndex >= 0)
		{
			check(!InstanceCullingInfos.IsAllocated(PrimitiveGroup.InstanceCullingInfoIndex));
			PrimitiveGroup.InstanceCullingInfoIndex = INDEX_NONE;
		}
	}
	else
	{
		check(PrimitiveGroup.InstanceCullingInfoIndex == INDEX_NONE);
	}
}

void FLumenSceneData::UpdatePrimitiveGroupCullingInfo(const FLumenPrimitiveGroup& PrimitiveGroup, const FRenderBounds& NewWorldBounds, bool bForcePrimitiveLevel, bool bAdditive)
{
	check(!bForcePrimitiveLevel || PrimitiveGroup.InstanceCullingInfoIndex >= 0);
	FLumenPrimitiveGroupCullingInfo& CullingInfo = GetPrimitiveGroupCullingInfo(PrimitiveGroup, bForcePrimitiveLevel);
	
	if (bAdditive)
	{
		CullingInfo.WorldSpaceBoundingBox += NewWorldBounds;
	}
	else
	{
		CullingInfo.WorldSpaceBoundingBox = NewWorldBounds;
	}

	if (!bForcePrimitiveLevel)
	{
		CullingInfo.bVisible = PrimitiveGroup.MeshCardsIndex >= 0;
		CullingInfo.bValidMeshCards = PrimitiveGroup.HasValidMeshCards();
	}
}

class FLumenMergedMeshCards
{
public:
	FLumenMergedMeshCards()
	{
		MergedBounds.Init();

		for (int32 AxisAlignedDirectionIndex = 0; AxisAlignedDirectionIndex < Lumen::NumAxisAlignedDirections; ++AxisAlignedDirectionIndex)
		{
			InstanceCardAreaPerDirection[AxisAlignedDirectionIndex] = 0;
		}
	}

	void AddInstance(FBox InstanceBox, FMatrix InstanceToMerged, const FMeshCardsBuildData& MeshCardsBuildData)
	{
		MergedBounds += InstanceBox.TransformBy(InstanceToMerged);

		for (const FLumenCardBuildData& CardBuildData : MeshCardsBuildData.CardBuildData)
		{
			const FVector3f AxisX = FVector4f(InstanceToMerged.TransformVector((FVector)CardBuildData.OBB.AxisX));
			const FVector3f AxisY = FVector4f(InstanceToMerged.TransformVector((FVector)CardBuildData.OBB.AxisY));
			const FVector3f AxisZ = FVector4f(InstanceToMerged.TransformVector((FVector)CardBuildData.OBB.AxisZ));
			const FVector3f Extent = CardBuildData.OBB.Extent * FVector3f(AxisX.Length(), AxisY.Length(), AxisZ.Length());

			const float InstanceCardArea = Extent.X * Extent.Y;
			const FVector3f CardDirection = AxisZ.GetUnsafeNormal();

			for (int32 AxisAlignedDirectionIndex = 0; AxisAlignedDirectionIndex < Lumen::NumAxisAlignedDirections; ++AxisAlignedDirectionIndex)
			{
				const FVector3f AxisDirection = MeshCardRepresentation::GetAxisAlignedDirection(AxisAlignedDirectionIndex);
				const float AxisProjection = CardDirection.Dot(AxisDirection);

				if (AxisProjection > 0.0f)
				{
					InstanceCardAreaPerDirection[AxisAlignedDirectionIndex] += AxisProjection * InstanceCardArea;
				}
			}
		}
	}

	FBox MergedBounds;
	float InstanceCardAreaPerDirection[Lumen::NumAxisAlignedDirections];
};

void BuildMeshCardsDataForHeightfield(const FLumenPrimitiveGroup& PrimitiveGroup, FMeshCardsBuildData& MeshCardsBuildData, FMatrix& MeshCardsLocalToWorld)
{
	const FPrimitiveSceneProxy* Proxy = PrimitiveGroup.GetPrimitives()[0]->Proxy;

	MeshCardsLocalToWorld = Proxy->GetLocalToWorld();

	// Make sure that the card isn't placed directly on the geometry
	const FVector BoundsMargin = FVector(CVarLumenSurfaceCacheHeightfieldCaptureMargin.GetValueOnRenderThread()) / MeshCardsLocalToWorld.GetScaleVector();

	MeshCardsBuildData.Bounds = Proxy->GetLocalBounds().GetBox().ExpandBy(BoundsMargin);

	// Add a single top down card
	MeshCardsBuildData.CardBuildData.SetNum(1);
	{
		FLumenCardBuildData& CardBuildData = MeshCardsBuildData.CardBuildData[0];

		// Set rotation
		uint32 AxisAlignedDirectionIndex = 5;
		CardBuildData.OBB.AxisZ = MeshCardRepresentation::GetAxisAlignedDirection(AxisAlignedDirectionIndex);
		CardBuildData.OBB.AxisZ.FindBestAxisVectors(CardBuildData.OBB.AxisX, CardBuildData.OBB.AxisY);
		CardBuildData.OBB.AxisX = FVector3f::CrossProduct(CardBuildData.OBB.AxisZ, CardBuildData.OBB.AxisY);
		CardBuildData.OBB.AxisX.Normalize();

		CardBuildData.OBB.Origin = (FVector3f)MeshCardsBuildData.Bounds.GetCenter();
		CardBuildData.OBB.Extent = CardBuildData.OBB.RotateLocalToCard((FVector3f)MeshCardsBuildData.Bounds.GetExtent()).GetAbs();

		CardBuildData.AxisAlignedDirectionIndex = AxisAlignedDirectionIndex;
	}
}

void BuildMeshCardsDataForMergedInstances(const FLumenPrimitiveGroup& PrimitiveGroup, FMeshCardsBuildData& MeshCardsBuildData, FMatrix& MeshCardsLocalToWorld)
{
	MeshCardsLocalToWorld.SetIdentity();

	// Pick first largest bbox as a reference frame
	float LargestInstanceArea = -1.0f;
	for (const FPrimitiveSceneInfo* PrimitiveSceneInfo : PrimitiveGroup.GetPrimitives())
	{
		const FMatrix& PrimitiveToWorld = PrimitiveSceneInfo->Proxy->GetLocalToWorld();
		const FBoxSphereBounds& PrimitiveBounds = PrimitiveSceneInfo->Proxy->GetBounds();
		float InstanceArea = BoxSurfaceArea(PrimitiveBounds.BoxExtent);
		FMatrix InstanceMeshCardsLocalToWorld = PrimitiveToWorld;

		if (const FInstanceSceneDataBuffers *InstanceSceneData = PrimitiveSceneInfo->GetInstanceSceneDataBuffers())
		{
			// Instance data must be available on CPU.
			check(!InstanceSceneData->IsInstanceDataGPUOnly());

			for (int32 InstanceIndex = 0; InstanceIndex < InstanceSceneData->GetNumInstances(); ++InstanceIndex)
			{
				InstanceArea = BoxSurfaceArea((FVector)InstanceSceneData->GetInstanceLocalBounds(InstanceIndex).GetExtent());
				InstanceMeshCardsLocalToWorld = InstanceSceneData->GetInstanceToWorld(InstanceIndex);
			}
		}
		if (InstanceArea > LargestInstanceArea)
		{
			MeshCardsLocalToWorld = InstanceMeshCardsLocalToWorld;
			LargestInstanceArea = InstanceArea;
		}
	}

	const FMatrix WorldToMeshCardsLocal = MeshCardsLocalToWorld.Inverse();

	MeshCardsBuildData.Bounds.Init();

	FLumenMergedMeshCards MergedMeshCards;

	for (const FPrimitiveSceneInfo* PrimitiveSceneInfo : PrimitiveGroup.GetPrimitives())
	{
		const FCardRepresentationData* CardRepresentationData = PrimitiveSceneInfo->Proxy->GetMeshCardRepresentation();

		if (CardRepresentationData)
		{
			const FMatrix& PrimitiveToWorld = PrimitiveSceneInfo->Proxy->GetLocalToWorld();
			const FMeshCardsBuildData& PrimitiveMeshCardsBuildData = CardRepresentationData->MeshCardsBuildData;
			const FMatrix PrimitiveLocalToMeshCardsLocal = PrimitiveToWorld * WorldToMeshCardsLocal;

			if (const FInstanceSceneDataBuffers *InstanceSceneData = PrimitiveSceneInfo->GetInstanceSceneDataBuffers())
			{
				// Instance data must be available on CPU.
				check(!InstanceSceneData->IsInstanceDataGPUOnly());

				for (int32 InstanceIndex = 0; InstanceIndex < InstanceSceneData->GetNumInstances(); ++InstanceIndex)
				{
					FMatrix InstanceToWorld = InstanceSceneData->GetInstanceToWorld(InstanceIndex);
					MergedMeshCards.AddInstance(
						InstanceSceneData->GetInstanceLocalBounds(InstanceIndex).ToBox(),
						InstanceToWorld * WorldToMeshCardsLocal,
						PrimitiveMeshCardsBuildData);
				}
			}
			else
			{
				MergedMeshCards.AddInstance(
					PrimitiveSceneInfo->Proxy->GetLocalBounds().GetBox(),
					PrimitiveLocalToMeshCardsLocal,
					PrimitiveMeshCardsBuildData);
			}
		}
	}

	// Spawn cards only on faces passing min area threshold
	TArray<int32, TInlineAllocator<Lumen::NumAxisAlignedDirections>> AxisAlignedDirectionsToSpawnCards;
	for (int32 AxisAlignedDirectionIndex = 0; AxisAlignedDirectionIndex < Lumen::NumAxisAlignedDirections; ++AxisAlignedDirectionIndex)
	{
		FVector3f MergedExtent = (FVector3f)MergedMeshCards.MergedBounds.GetExtent();
		MergedExtent[AxisAlignedDirectionIndex / 2] = 1.0f;
		const float MergedFaceArea = MergedExtent.X * MergedExtent.Y * MergedExtent.Z;

		if (MergedMeshCards.InstanceCardAreaPerDirection[AxisAlignedDirectionIndex] > GLumenMeshCardsMergedCardMinSurfaceArea * MergedFaceArea)
		{
			AxisAlignedDirectionsToSpawnCards.Add(AxisAlignedDirectionIndex);
		}
	}

	if (MergedMeshCards.MergedBounds.IsValid && AxisAlignedDirectionsToSpawnCards.Num() > 0)
	{
		// Make sure BBox isn't empty and we can generate card representation for it. This handles e.g. infinitely thin planes.
		const FVector SafeCenter = MergedMeshCards.MergedBounds.GetCenter();
		const FVector SafeExtent = FVector::Max(MergedMeshCards.MergedBounds.GetExtent() + 1.0f, FVector(5.0f));
		const FBox SafeMergedBounds = FBox(SafeCenter - SafeExtent, SafeCenter + SafeExtent);

		MeshCardsBuildData.Bounds = SafeMergedBounds;

		MeshCardsBuildData.CardBuildData.SetNum(AxisAlignedDirectionsToSpawnCards.Num());
		uint32 CardBuildDataIndex = 0;

		for (int32 AxisAlignedDirectionIndex : AxisAlignedDirectionsToSpawnCards)
		{
			FLumenCardBuildData& CardBuildData = MeshCardsBuildData.CardBuildData[CardBuildDataIndex];
			++CardBuildDataIndex;

			// Set rotation
			CardBuildData.OBB.AxisZ = MeshCardRepresentation::GetAxisAlignedDirection(AxisAlignedDirectionIndex);
			CardBuildData.OBB.AxisZ.FindBestAxisVectors(CardBuildData.OBB.AxisX, CardBuildData.OBB.AxisY);
			CardBuildData.OBB.AxisX = FVector3f::CrossProduct(CardBuildData.OBB.AxisZ, CardBuildData.OBB.AxisY);
			CardBuildData.OBB.AxisX.Normalize();

			CardBuildData.OBB.Origin = (FVector3f)SafeMergedBounds.GetCenter();	// LWC_TODO: Precision Loss
			CardBuildData.OBB.Extent = CardBuildData.OBB.RotateLocalToCard((FVector3f)SafeMergedBounds.GetExtent() + FVector3f(1.0f)).GetAbs();

			CardBuildData.AxisAlignedDirectionIndex = AxisAlignedDirectionIndex;
		}
	}
}

void FLumenSceneData::AddMeshCards(int32 PrimitiveGroupIndex)
{
	FLumenPrimitiveGroup& PrimitiveGroup = PrimitiveGroups[PrimitiveGroupIndex];

	if (PrimitiveGroup.MeshCardsIndex < 0)
	{
		if (PrimitiveGroup.IsHeightfield())
		{
			// Landscape component handling
			FMatrix LocalToWorld;
			FMeshCardsBuildData MeshCardsBuildData;
			BuildMeshCardsDataForHeightfield(PrimitiveGroup, MeshCardsBuildData, LocalToWorld);

			AddMeshCardsFromBuildData(PrimitiveGroupIndex, LocalToWorld, MeshCardsBuildData, PrimitiveGroup);
		}
		else if (PrimitiveGroup.HasMergedInstances())
		{
			// Multiple meshes merged together
			FMatrix LocalToWorld;
			FMeshCardsBuildData MeshCardsBuildData;
			BuildMeshCardsDataForMergedInstances(PrimitiveGroup, MeshCardsBuildData, LocalToWorld);

			AddMeshCardsFromBuildData(PrimitiveGroupIndex, LocalToWorld, MeshCardsBuildData, PrimitiveGroup);
		}
		else
		{
			// Single mesh
			ensure(PrimitiveGroup.GetPrimitives().Num() == 1);
			const FPrimitiveSceneInfo* PrimitiveSceneInfo = PrimitiveGroup.GetPrimitives()[0];

			const FMatrix LocalToWorld = PrimitiveSceneInfo->Proxy->GetMeshCardsToWorld(PrimitiveGroup.PrimitiveInstanceIndex);

			const FCardRepresentationData* CardRepresentationData = PrimitiveSceneInfo->Proxy->GetMeshCardRepresentation();
			if (CardRepresentationData)
			{
				const FMeshCardsBuildData& MeshCardsBuildData = CardRepresentationData->MeshCardsBuildData;
				AddMeshCardsFromBuildData(PrimitiveGroupIndex, LocalToWorld, MeshCardsBuildData, PrimitiveGroup);
			}
		}

		FLumenPrimitiveGroupCullingInfo& CullingInfo = GetPrimitiveGroupCullingInfo(PrimitiveGroup);
		check(!CullingInfo.bVisible && CullingInfo.bValidMeshCards);

		if (PrimitiveGroup.MeshCardsIndex >= 0)
		{
			// Copy ScenePrimitive->GetIndex() in order to prevent from deferencing possibly deleted ScenePrimitive*
			FLumenMeshCards& MeshCardsInstance = MeshCards[PrimitiveGroup.MeshCardsIndex];

			MeshCardsInstance.ScenePrimitiveIndices.Reset();
			MeshCardsInstance.ScenePrimitiveIndices.Reserve(PrimitiveGroup.GetPrimitives().Num());

			for (const FPrimitiveSceneInfo* ScenePrimitive : PrimitiveGroup.GetPrimitives())
			{
				if (ScenePrimitive->IsIndexValid())
				{
					MeshCardsInstance.ScenePrimitiveIndices.Add(ScenePrimitive->GetIndex());
					PrimitivesToUpdateMeshCards.Add(ScenePrimitive->GetIndex());
				}
			}

			CullingInfo.bVisible = true;
		}
		else
		{
			// Can't spawn mesh cards, mark this primitive as invalid
			PrimitiveGroup.SetHasValidMeshCards(false);
			UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);
			CullingInfo.bValidMeshCards = false;
		}
	}
}

bool IsMatrixOrthogonal(const FMatrix& Matrix)
{
	const FVector MatrixScale = Matrix.GetScaleVector();

	if (MatrixScale.GetAbsMin() >= KINDA_SMALL_NUMBER)
	{
		FVector AxisX;
		FVector AxisY;
		FVector AxisZ;
		Matrix.GetUnitAxes(AxisX, AxisY, AxisZ);

		return FMath::Abs(AxisX | AxisY) < KINDA_SMALL_NUMBER
			&& FMath::Abs(AxisX | AxisZ) < KINDA_SMALL_NUMBER
			&& FMath::Abs(AxisY | AxisZ) < KINDA_SMALL_NUMBER;
	}

	return false;
}

bool MeshCardCullTest(const FLumenCardBuildData& CardBuildData, const FVector3f LocalToWorldScale, float MinFaceSurfaceArea, int32 CardIndex)
{
#if UE_BUILD_DEBUG || UE_BUILD_DEVELOPMENT
	if (GLumenMeshCardsDebugSingleCard >= 0)
	{
		return GLumenMeshCardsDebugSingleCard == CardIndex;
	}
#endif

	const FVector3f CardSpaceScale = CardBuildData.OBB.RotateLocalToCard(LocalToWorldScale).GetAbs();
	const FVector3f ScaledBoundsSize = 2.0f * CardBuildData.OBB.Extent * CardSpaceScale;
	const float SurfaceArea = ScaledBoundsSize.X * ScaledBoundsSize.Y;
	const bool bCardPassedCulling = (!GLumenMeshCardsCullFaces || SurfaceArea > MinFaceSurfaceArea);

	return bCardPassedCulling;
}

void FLumenSceneData::AddMeshCardsFromBuildData(int32 PrimitiveGroupIndex, const FMatrix& LocalToWorld, const FMeshCardsBuildData& MeshCardsBuildData, FLumenPrimitiveGroup& PrimitiveGroup)
{
	PrimitiveGroup.MeshCardsIndex = -1;
	PrimitiveGroup.HeightfieldIndex = -1;

	const FVector3f LocalToWorldScale = (FVector3f)LocalToWorld.GetScaleVector();
	const FVector3f ScaledBoundSize = (FVector3f)MeshCardsBuildData.Bounds.GetSize() * LocalToWorldScale;
	const FVector3f FaceSurfaceArea(ScaledBoundSize.Y * ScaledBoundSize.Z, ScaledBoundSize.X * ScaledBoundSize.Z, ScaledBoundSize.Y * ScaledBoundSize.X);
	const float LargestFaceArea = FaceSurfaceArea.GetMax();
	const float MinFaceSurfaceArea = LumenMeshCards::GetCardMinSurfaceArea(PrimitiveGroup.IsEmissiveLightSource());

	if (LargestFaceArea > MinFaceSurfaceArea
		&& IsMatrixOrthogonal(LocalToWorld)) // #lumen_todo: implement card capture for non orthogonal local to world transforms
	{
		const int32 NumBuildDataCards = MeshCardsBuildData.CardBuildData.Num();

		uint32 NumCards = 0;

		for (int32 CardIndexInBuildData = 0; CardIndexInBuildData < NumBuildDataCards; ++CardIndexInBuildData)
		{
			const FLumenCardBuildData& CardBuildData = MeshCardsBuildData.CardBuildData[CardIndexInBuildData];

			if (MeshCardCullTest(CardBuildData, LocalToWorldScale, MinFaceSurfaceArea, CardIndexInBuildData))
			{
				++NumCards;
			}
		}

		if (NumCards > 0)
		{
			const int32 FirstCardIndex = Cards.AddSpan(NumCards);
			const int32 FirstCardCullingInfoIndex = CardCullingInfos.AddSpan(NumCards);
			check(FirstCardIndex == FirstCardCullingInfoIndex);

			const int32 MeshCardsIndex = MeshCards.AddSpan(1);
			PrimitiveGroup.MeshCardsIndex = MeshCardsIndex;
			FLumenMeshCards& MeshCardsInstance = MeshCards[MeshCardsIndex];
			MeshCardsInstance.Initialize(
				LocalToWorld,
				PrimitiveGroupIndex,
				FirstCardIndex,
				NumCards,
				MeshCardsBuildData,
				PrimitiveGroup);

			MeshCardsIndicesToUpdateInBuffer.Add(MeshCardsIndex);

			if (PrimitiveGroup.IsHeightfield())
			{
				const int32 HeightfieldIndex = Heightfields.AddSpan(1);
				PrimitiveGroup.HeightfieldIndex = HeightfieldIndex;
				Heightfields[HeightfieldIndex].Initialize(MeshCardsIndex);

				HeightfieldIndicesToUpdateInBuffer.Add(HeightfieldIndex);
			}

			// Add cards
			int32 LocalCardIndex = 0;
			for (int32 CardIndexInBuildData = 0; CardIndexInBuildData < NumBuildDataCards; ++CardIndexInBuildData)
			{	
				const FLumenCardBuildData& CardBuildData = MeshCardsBuildData.CardBuildData[CardIndexInBuildData];

				if (MeshCardCullTest(CardBuildData, LocalToWorldScale, MinFaceSurfaceArea, CardIndexInBuildData))
				{
					const int32 CardInsertIndex = FirstCardIndex + LocalCardIndex;
					FLumenCard& Card = Cards[CardInsertIndex];
					FLumenCardCullingInfo& CardCullingInfo = CardCullingInfos[CardInsertIndex];

					Card.Initialize(
						PrimitiveGroup.CustomId,
						LocalToWorld,
						MeshCardsInstance,
						CardBuildData,
						LocalCardIndex,
						MeshCardsIndex,
						CardIndexInBuildData);

					CardCullingInfo.Initialize(Card);

					CardIndicesToUpdateInBuffer.Add(CardInsertIndex);

					++LocalCardIndex;
				}
			}

			MeshCardsInstance.UpdateLookup(Cards);

			UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);
		}
	}
}

void FLumenSceneData::RemoveMeshCards(int32 PrimitiveGroupIndex, bool bUpdateCullingInfo)
{
	FLumenPrimitiveGroup& PrimitiveGroup = PrimitiveGroups[PrimitiveGroupIndex];

	if (PrimitiveGroup.MeshCardsIndex >= 0)
	{
		FLumenMeshCards& MeshCardsInstance = MeshCards[PrimitiveGroup.MeshCardsIndex];

		for (uint32 CardIndex = MeshCardsInstance.FirstCardIndex; CardIndex < MeshCardsInstance.FirstCardIndex + MeshCardsInstance.NumCards; ++CardIndex)
		{
			RemoveCardFromAtlas(CardIndex);
		}

		if (PrimitiveGroup.HeightfieldIndex >= 0)
		{
			Heightfields.RemoveSpan(PrimitiveGroup.HeightfieldIndex, 1);
			HeightfieldIndicesToUpdateInBuffer.Add(PrimitiveGroup.HeightfieldIndex);
		}

		// Update surface cache mapping
		for (int32 ScenePrimitiveIndex : MeshCardsInstance.ScenePrimitiveIndices)
		{
			PrimitivesToUpdateMeshCards.Add(ScenePrimitiveIndex);
		}
		MeshCardsInstance.ScenePrimitiveIndices.Reset();

		Cards.RemoveSpan(MeshCardsInstance.FirstCardIndex, MeshCardsInstance.NumCards);
		CardCullingInfos.RemoveSpan(MeshCardsInstance.FirstCardIndex, MeshCardsInstance.NumCards);
		MeshCards.RemoveSpan(PrimitiveGroup.MeshCardsIndex, 1);

		MeshCardsIndicesToUpdateInBuffer.Add(PrimitiveGroup.MeshCardsIndex);

		PrimitiveGroup.MeshCardsIndex = -1;
		PrimitiveGroup.HeightfieldIndex = -1;

		UpdatePrimitiveGroupGPUBufferEntry(PrimitiveGroupIndex);

		if (bUpdateCullingInfo)
		{
			FLumenPrimitiveGroupCullingInfo& CullingInfo = GetPrimitiveGroupCullingInfo(PrimitiveGroup);
			check(CullingInfo.bVisible && CullingInfo.bValidMeshCards);
			CullingInfo.bVisible = false;
		}
	}
}

void FLumenSceneData::UpdateMeshCards(const FMatrix& LocalToWorld, int32 MeshCardsIndex, const FMeshCardsBuildData& MeshCardsBuildData)
{
	if (MeshCardsIndex >= 0 && IsMatrixOrthogonal(LocalToWorld))
	{
		FLumenMeshCards& MeshCardsInstance = MeshCards[MeshCardsIndex];
		MeshCardsInstance.SetTransform(LocalToWorld);
		MeshCardsIndicesToUpdateInBuffer.Add(MeshCardsIndex);

		for (uint32 LocalCardIndex = 0; LocalCardIndex < MeshCardsInstance.NumCards; ++LocalCardIndex)
		{
			const uint32 CardIndex = MeshCardsInstance.FirstCardIndex + LocalCardIndex;
			FLumenCard& Card = Cards[CardIndex];
			FLumenCardCullingInfo& CardCullingInfo = CardCullingInfos[CardIndex];

			Card.SetTransform(LocalToWorld, MeshCardsInstance);
			CardCullingInfo.SetTransform(Card);

			CardIndicesToUpdateInBuffer.Add(CardIndex);
		}
	}
}

void FLumenSceneData::InvalidateSurfaceCache(FRHIGPUMask GPUMask, int32 MeshCardsIndex)
{
	if (MeshCardsIndex >= 0)
	{
		FLumenMeshCards& MeshCardsInstance = MeshCards[MeshCardsIndex];
		for (uint32 CardIndex = MeshCardsInstance.FirstCardIndex; CardIndex < MeshCardsInstance.FirstCardIndex + MeshCardsInstance.NumCards; ++CardIndex)
		{
			const FLumenCard& LumenCard = Cards[CardIndex];
			for (int32 ResLevel = LumenCard.MinAllocatedResLevel; ResLevel <= LumenCard.MaxAllocatedResLevel; ++ResLevel)
			{
				const FLumenSurfaceMipMap& MipMap = LumenCard.GetMipMap(ResLevel);
				if (MipMap.IsAllocated())
				{
					for (int32 LocalPageIndex = 0; LocalPageIndex < int32(MipMap.SizeInPagesX * MipMap.SizeInPagesY); ++LocalPageIndex)
					{
						const int32 PageIndex = MipMap.GetPageTableIndex(LocalPageIndex);
						if (GetPageTableEntry(PageIndex).IsMapped())
						{
							for (uint32 GPUIndex : GPUMask)
							{
								if (PagesToRecaptureHeap[GPUIndex].IsPresent(PageIndex))
								{
									PagesToRecaptureHeap[GPUIndex].Update(GetSurfaceCacheUpdateFrameIndex(), PageIndex);
								}
								else
								{
									PagesToRecaptureHeap[GPUIndex].Add(GetSurfaceCacheUpdateFrameIndex(), PageIndex);
								}								
							}
						}
					}
				}
			}
		}
	}
}

void FLumenSceneData::RemoveCardFromAtlas(int32 CardIndex)
{
	FLumenCard& Card = Cards[CardIndex];
	FLumenCardCullingInfo& CardCullingInfo = CardCullingInfos[CardIndex];
	NumDesiredSurfaceCacheTexels += CardCullingInfo.SetDesiredLockedResLevel(0, Card);
	check(NumDesiredSurfaceCacheTexels >= 0);
	CardCullingInfo.DesiredLockedResLevelOnLastAlloc = 0;
	FreeVirtualSurface(Card, Card.MinAllocatedResLevel, Card.MaxAllocatedResLevel);
	CardIndicesToUpdateInBuffer.Add(CardIndex);
}

void FLumenCardCullingInfo::Initialize(const FLumenCard& Card)
{
	SetTransform(Card);
}

void FLumenCardCullingInfo::SetTransform(const FLumenCard& Card)
{
	const FLumenCardOBBf& MeshCardsOBB = Card.GetMeshCardsOBB();
	AxisZIndex = Card.LocalOBBAxisZ / 2;
	MeshCardsSpaceCenter = MeshCardsOBB.Origin;
	MeshCardsSpaceExtent = MeshCardsOBB.Extent;
}

float FLumenCardCullingInfo::GetMaxCardExtent() const
{
	switch (AxisZIndex)
	{
	case 0u:
		return FMath::Max(MeshCardsSpaceExtent.Y, MeshCardsSpaceExtent.Z);
	case 1u:
		return FMath::Max(MeshCardsSpaceExtent.Z, MeshCardsSpaceExtent.X);
	case 2u:
		return FMath::Max(MeshCardsSpaceExtent.X, MeshCardsSpaceExtent.Y);
	default:
		checkNoEntry();
		return 0.0f;
	}
}

int32 FLumenCardCullingInfo::SetDesiredLockedResLevel(int32 NewLevel, const FLumenCard& Card)
{
	const int32 OldLevel = DesiredLockedResLevel;
	int32 DesiredTexelsDelta = 0;

	if (OldLevel != NewLevel)
	{
		DesiredLockedResLevel = NewLevel;

#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
		if (OldLevel > 0)
		{
			FLumenMipMapDesc MipDesc;
			Card.GetMipMapDesc(OldLevel, MipDesc);

			const FIntPoint MipSize = MipDesc.SizeInPages * MipDesc.PageResolution;
			DesiredTexelsDelta -= MipSize.X * MipSize.Y;
		}

		if (NewLevel > 0)
		{
			FLumenMipMapDesc MipDesc;
			Card.GetMipMapDesc(NewLevel, MipDesc);

			const FIntPoint MipSize = MipDesc.SizeInPages * MipDesc.PageResolution;
			DesiredTexelsDelta += MipSize.X * MipSize.Y;
		}
#endif
	}

	return DesiredTexelsDelta;
}

FLumenCard::~FLumenCard()
{
	for (int32 MipIndex = 0; MipIndex < UE_ARRAY_COUNT(SurfaceMipMaps); ++MipIndex)
	{
		ensure(SurfaceMipMaps[MipIndex].PageTableSpanSize == 0);
	}
}

void FLumenCard::Initialize(
	uint32 CustomId,
	const FMatrix& LocalToWorld,
	const FLumenMeshCards& InMeshCardsInstance,
	const FLumenCardBuildData& CardBuildData,
	int32 InIndexInMeshCards,
	int32 InMeshCardsIndex,
	uint8 InIndexInBuildData)
{
	check(CardBuildData.AxisAlignedDirectionIndex < Lumen::NumAxisAlignedDirections);
	check(InIndexInMeshCards >= 0 && InIndexInMeshCards < 32);

	SetLocalOBB(CardBuildData.OBB);
	IndexInMeshCards = InIndexInMeshCards;
	MeshCardsIndex = InMeshCardsIndex;
	IndexInBuildData = InIndexInBuildData;
	AxisAlignedDirectionIndex = CardBuildData.AxisAlignedDirectionIndex;
	bHeightfield = InMeshCardsInstance.bHeightfield;
	bFarField = InMeshCardsInstance.bFarField;
	DilationMode = uint32(CardBuildData.DilationMode);

	SetTransform(LocalToWorld, InMeshCardsInstance);

	const FVector3f WorldOBBExtent = GetWorldOBB().Extent;
	const float CardAspect = WorldOBBExtent.X / WorldOBBExtent.Y;
	ComputeAndSetResLevelXYBias(CardAspect);

	if (CustomId != UINT32_MAX)
	{
		const FIntPoint& ResLevelXYBias = ResLevelToResLevelXYBias();
		CardSharingId = FLumenCardId(CustomId, AxisAlignedDirectionIndex, ResLevelXYBias.X, ResLevelXYBias.Y);
	}
}

void FLumenCard::SetTransform(const FMatrix& LocalToWorld, const FLumenMeshCards& MeshCards)
{
	const FLumenCardOBBf LocalOBB = GetLocalOBB();

	bool bLocalAxisXFlipped;
	const auto WorldOBB = LocalOBB.Transform(LocalToWorld, &bLocalAxisXFlipped);

	bAxisXFlipped = bLocalAxisXFlipped;
	WorldOBBAxisX = WorldOBB.AxisX;
	WorldOBBAxisY = WorldOBB.AxisY;
	WorldOBBAxisZ = WorldOBB.AxisZ;
	WorldOBBOrigin = WorldOBB.Origin;
	LocalToWorldScale = MeshCards.LocalToWorldScale;

	const float Epsilon = 0.001f;
	check(FVector3f(FVector(MeshCards.WorldToLocalRotation.TransformVector(FVector(WorldOBB.AxisX)))).Equals(GetMeshCardsOBB().AxisX, Epsilon));
	check(FVector3f(FVector(MeshCards.WorldToLocalRotation.TransformVector(FVector(WorldOBB.AxisY)))).Equals(GetMeshCardsOBB().AxisY, Epsilon));
	check(FVector3f(FVector(MeshCards.WorldToLocalRotation.TransformVector(FVector(WorldOBB.AxisZ)))).Equals(GetMeshCardsOBB().AxisZ, Epsilon));
	check(WorldOBB.Extent.Equals(GetWorldOBB().Extent, FMath::Max(GetWorldOBB().Extent.GetMax() * 1e-6f, Epsilon)));
}

void FLumenCard::UpdateMinMaxAllocatedLevel()
{
	MinAllocatedResLevel = UINT8_MAX;
	MaxAllocatedResLevel = 0;

	for (int32 ResLevelIndex = Lumen::MinResLevel; ResLevelIndex <= Lumen::MaxResLevel; ++ResLevelIndex)
	{
		if (GetMipMap(ResLevelIndex).IsAllocated())
		{
			MinAllocatedResLevel = FMath::Min<int32>(MinAllocatedResLevel, ResLevelIndex);
			MaxAllocatedResLevel = FMath::Max<int32>(MaxAllocatedResLevel, ResLevelIndex);
		}
	}
}

FIntPoint FLumenCard::ResLevelToResLevelXYBias() const
{
	return FIntPoint(PackedResLevelXYBias & 0xFu, PackedResLevelXYBias >> 4u);
}

void FLumenCard::ComputeAndSetResLevelXYBias(float CardAspect)
{
	uint32 BiasX = 0;
	uint32 BiasY = 0;

	// ResLevel bias to account for card's aspect
	if (CardAspect >= 1.0f)
	{
		BiasY = FMath::FloorLog2(FMath::RoundToInt(CardAspect));
	}
	else
	{
		BiasX = FMath::FloorLog2(FMath::RoundToInt(1.0f / CardAspect));
	}

	static_assert(Lumen::MaxResLevel - Lumen::MinResLevel < 16u);
	BiasX = FMath::Clamp(BiasX, 0u, Lumen::MaxResLevel - Lumen::MinResLevel);
	BiasY = FMath::Clamp(BiasY, 0u, Lumen::MaxResLevel - Lumen::MinResLevel);
	
	PackedResLevelXYBias = (BiasY << 4u) | BiasX;
}

void FLumenCard::GetMipMapDesc(int32 ResLevel, FLumenMipMapDesc& Desc) const
{
	check(ResLevel >= Lumen::MinResLevel && ResLevel <= Lumen::MaxResLevel);

	const FIntPoint ResLevelBias = ResLevelToResLevelXYBias();
	Desc.ResLevelX = FMath::Clamp<int32>(ResLevel - ResLevelBias.X, (int32)Lumen::MinResLevel, (int32)Lumen::MaxResLevel);
	Desc.ResLevelY = FMath::Clamp<int32>(ResLevel - ResLevelBias.Y, (int32)Lumen::MinResLevel, (int32)Lumen::MaxResLevel);

	// Allocations which exceed a physical page are aligned to multiples of a virtual page to maximize atlas usage
	if (Desc.ResLevelX > Lumen::SubAllocationResLevel || Desc.ResLevelY > Lumen::SubAllocationResLevel)
	{
		// Clamp res level to page size
		Desc.ResLevelX = FMath::Max<int32>(Desc.ResLevelX, Lumen::SubAllocationResLevel);
		Desc.ResLevelY = FMath::Max<int32>(Desc.ResLevelY, Lumen::SubAllocationResLevel);

		Desc.bSubAllocation = false;
		Desc.SizeInPages.X = 1u << (Desc.ResLevelX - Lumen::SubAllocationResLevel);
		Desc.SizeInPages.Y = 1u << (Desc.ResLevelY - Lumen::SubAllocationResLevel);
		Desc.Resolution.X = Desc.SizeInPages.X * Lumen::VirtualPageSize;
		Desc.Resolution.Y = Desc.SizeInPages.Y * Lumen::VirtualPageSize;
		Desc.PageResolution.X = Lumen::PhysicalPageSize;
		Desc.PageResolution.Y = Lumen::PhysicalPageSize;
	}
	else
	{
		Desc.bSubAllocation = true;
		Desc.SizeInPages.X = 1;
		Desc.SizeInPages.Y = 1;
		Desc.Resolution.X = 1 << Desc.ResLevelX;
		Desc.Resolution.Y = 1 << Desc.ResLevelY;
		Desc.PageResolution.X = Desc.Resolution.X;
		Desc.PageResolution.Y = Desc.Resolution.Y;
	}
}

void FLumenCard::GetSurfaceStats(const FLumenCardCullingInfo& CardCullingInfo, const TSparseSpanArray<FLumenPageTableEntry>& PageTable, FSurfaceStats& Stats) const
{
	if (IsAllocated())
	{
		for (int32 ResLevelIndex = MinAllocatedResLevel; ResLevelIndex <= MaxAllocatedResLevel; ++ResLevelIndex)
		{
			const FLumenSurfaceMipMap& MipMap = GetMipMap(ResLevelIndex);

			if (MipMap.IsAllocated())
			{
				uint32 NumVirtualTexels = 0;
				uint32 NumPhysicalTexels = 0;

				for (int32 LocalPageIndex = 0; LocalPageIndex < int32(MipMap.SizeInPagesX * MipMap.SizeInPagesY); ++LocalPageIndex)
				{
					const int32 PageTableIndex = MipMap.GetPageTableIndex(LocalPageIndex);
					const FLumenPageTableEntry& PageTableEntry = PageTable[PageTableIndex];

					NumVirtualTexels += PageTableEntry.GetNumVirtualTexels();
					NumPhysicalTexels += PageTableEntry.GetNumPhysicalTexels();
				}

				Stats.NumVirtualTexels += NumVirtualTexels;
				Stats.NumPhysicalTexels += NumPhysicalTexels;

				if (MipMap.bLocked)
				{
					Stats.NumLockedVirtualTexels += NumVirtualTexels;
					Stats.NumLockedPhysicalTexels += NumPhysicalTexels;
				}
			}
		}

		if (CardCullingInfo.DesiredLockedResLevel > MinAllocatedResLevel)
		{
			Stats.DroppedResLevels += CardCullingInfo.DesiredLockedResLevel - MinAllocatedResLevel;
		}
	}
}

static FVector3f ShuffleAxesNoFlip(uint32 AxisX, uint32 AxisY, uint32 AxisZ, const FVector3f& In)
{
	const uint32 OpCode = (AxisZ / 2u << 4u) | (AxisY / 2u << 2u) | (AxisX / 2u);
	switch (OpCode)
	{
	case 0b100100:
		return FVector3f(In.X, In.Y, In.Z);
	case 0b011000:
		return FVector3f(In.X, In.Z, In.Y);
	case 0b100001:
		return FVector3f(In.Y, In.X, In.Z);
	case 0b001001:
		return FVector3f(In.Z, In.X, In.Y);
	case 0b010010:
		return FVector3f(In.Y, In.Z, In.X);
	case 0b000110:
		return FVector3f(In.Z, In.Y, In.X);
	default:
		checkNoEntry();
		return FVector3f::ZeroVector;
	}
}

static FVector3f ShuffleAxes(uint32 AxisX, uint32 AxisY, uint32 AxisZ, const FVector3f& In)
{
	FVector3f Tmp;
	Tmp.X = AxisX & 0x1 ? In.X : -In.X;
	Tmp.Y = AxisY & 0x1 ? In.Y : -In.Y;
	Tmp.Z = AxisZ & 0x1 ? In.Z : -In.Z;

	return ShuffleAxesNoFlip(AxisX, AxisY, AxisZ, Tmp);
}

FVector3f FLumenCard::RotateCardToLocal(const FVector3f& CardPosition) const
{
	const FVector3f LocalPosition = ShuffleAxes(LocalOBBAxisX, LocalOBBAxisY, LocalOBBAxisZ, CardPosition);
	check(GetLocalOBB().RotateCardToLocal(CardPosition) == LocalPosition);
	return LocalPosition;
}

FVector3f FLumenCard::RotateCardToLocalNoFlip(const FVector3f& CardPosition) const
{
	const FVector3f LocalPosition = ShuffleAxesNoFlip(LocalOBBAxisX, LocalOBBAxisY, LocalOBBAxisZ, CardPosition);
	check(GetLocalOBB().RotateCardToLocal(CardPosition).GetAbs() == LocalPosition.GetAbs());
	return LocalPosition;
}

FVector3f FLumenCard::RotateLocalToCard(const FVector3f& LocalPosition) const
{
	uint32 InvAxes[3];
	InvAxes[LocalOBBAxisX / 2u] = 0u + (LocalOBBAxisX & 0x1);
	InvAxes[LocalOBBAxisY / 2u] = 2u + (LocalOBBAxisY & 0x1);
	InvAxes[LocalOBBAxisZ / 2u] = 4u + (LocalOBBAxisZ & 0x1);

	const FVector3f CardPosition = ShuffleAxes(InvAxes[0], InvAxes[1], InvAxes[2], LocalPosition); //-V614
	check(GetLocalOBB().RotateLocalToCard(LocalPosition) == CardPosition);
	return CardPosition;
}

FVector3f FLumenCard::RotateLocalToCardNoFlip(const FVector3f& LocalPosition) const
{
	uint32 InvAxes[3];
	InvAxes[LocalOBBAxisX / 2u] = 0u;
	InvAxes[LocalOBBAxisY / 2u] = 2u;
	InvAxes[LocalOBBAxisZ / 2u] = 4u;

	const FVector3f CardPosition = ShuffleAxesNoFlip(InvAxes[0], InvAxes[1], InvAxes[2], LocalPosition); //-V614
	check(GetLocalOBB().RotateLocalToCard(LocalPosition).GetAbs() == CardPosition.GetAbs());
	return CardPosition;
}

void FLumenCard::SetLocalOBB(const FLumenCardOBBf& In)
{
	const int32 AxisX = MeshCardRepresentation::GetAxisAlignedDirectionIndex(In.AxisX);
	const int32 AxisY = MeshCardRepresentation::GetAxisAlignedDirectionIndex(In.AxisY);
	const int32 AxisZ = MeshCardRepresentation::GetAxisAlignedDirectionIndex(In.AxisZ);
	check(AxisX >= 0 && AxisX <= 5 && AxisY >= 0 && AxisY <= 5 && AxisZ >= 0 && AxisZ <= 5);
	check(AxisX / 2 != AxisY / 2 && AxisY / 2 != AxisZ / 2 && AxisZ / 2 != AxisX / 2);

	LocalOBBAxisX = AxisX;
	LocalOBBAxisY = AxisY;
	LocalOBBAxisZ = AxisZ;
	LocalOBBOrigin = In.Origin;
	LocalOBBExtent = In.Extent;
}

FLumenCardOBBf FLumenCard::GetLocalOBB() const
{
	FLumenCardOBBf LocalOBB;
	LocalOBB.Origin = LocalOBBOrigin;
	LocalOBB.AxisX = MeshCardRepresentation::GetAxisAlignedDirection(LocalOBBAxisX);
	LocalOBB.AxisY = MeshCardRepresentation::GetAxisAlignedDirection(LocalOBBAxisY);
	LocalOBB.AxisZ = MeshCardRepresentation::GetAxisAlignedDirection(LocalOBBAxisZ);
	LocalOBB.Extent = LocalOBBExtent;
	return LocalOBB;
}

FLumenCardOBBf FLumenCard::GetMeshCardsOBB() const
{
	FLumenCardOBBf MeshCardsOBB = GetLocalOBB();
	MeshCardsOBB.AxisX = bAxisXFlipped ? -MeshCardsOBB.AxisX : MeshCardsOBB.AxisX;
	MeshCardsOBB.Origin *= LocalToWorldScale;
	MeshCardsOBB.Extent = RotateCardToLocalNoFlip(MeshCardsOBB.Extent) * LocalToWorldScale;
	return MeshCardsOBB;
}

FLumenCardOBBd FLumenCard::GetWorldOBB() const
{
	FLumenCardOBBd WorldOBB;
	WorldOBB.Origin = WorldOBBOrigin;
	WorldOBB.AxisX = WorldOBBAxisX;
	WorldOBB.AxisY = WorldOBBAxisY;
	WorldOBB.AxisZ = WorldOBBAxisZ;
	WorldOBB.Extent = LocalOBBExtent * RotateLocalToCardNoFlip(LocalToWorldScale);
	WorldOBB.Extent.Z = FMath::Max(WorldOBB.Extent.Z, 1.0f);
	return WorldOBB;
}

void FLumenMeshCards::Initialize(
	const FMatrix& InLocalToWorld,
	int32 InPrimitiveGroupIndex,
	uint32 InFirstCardIndex,
	uint32 InNumCards,
	const FMeshCardsBuildData& MeshCardsBuildData,
	const FLumenPrimitiveGroup& PrimitiveGroup)
{
	PrimitiveGroupIndex = InPrimitiveGroupIndex;

	LocalBounds = MeshCardsBuildData.Bounds;
	bMostlyTwoSided = MeshCardsBuildData.bMostlyTwoSided;

	ResolutionScale = PrimitiveGroup.CardResolutionScale;
	FirstCardIndex = InFirstCardIndex;
	NumCards = InNumCards;

	bFarField = PrimitiveGroup.IsFarField();
	bHeightfield = PrimitiveGroup.IsHeightfield();
	bEmissiveLightSource = PrimitiveGroup.IsEmissiveLightSource();

	SetTransform(InLocalToWorld);
}

void FLumenMeshCards::UpdateLookup(const TSparseSpanArray<FLumenCard>& Cards)
{
	for (int32 AxisAlignedDirectionIndex = 0; AxisAlignedDirectionIndex < Lumen::NumAxisAlignedDirections; ++AxisAlignedDirectionIndex)
	{
		CardLookup[AxisAlignedDirectionIndex] = 0;
	}

	for (uint32 LocalCardIndex = 0; LocalCardIndex < NumCards; ++LocalCardIndex)
	{
		const uint32 CardIndex = FirstCardIndex + LocalCardIndex;
		const FLumenCard& Card = Cards[CardIndex];
		
		const uint32 BitMask = (1 << LocalCardIndex);
		CardLookup[Card.AxisAlignedDirectionIndex] |= BitMask;
	}
}

void FLumenMeshCards::SetTransform(const FMatrix& InLocalToWorld)
{
	LocalToWorld = InLocalToWorld;
	LocalToWorldScale = FVector3f(LocalToWorld.GetScaleVector());

	WorldToLocalRotation = LocalToWorld;
	WorldToLocalRotation.RemoveScaling();
	WorldToLocalRotation.SetOrigin(FVector::ZeroVector);
	WorldToLocalRotation = WorldToLocalRotation.GetTransposed();
}