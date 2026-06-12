// Copyright Epic Games, Inc. All Rights Reserved.

#include "CurveSkinningSceneExtension.h"
#include "CurveSkinning/CurveSkinningSceneExtensionInterface.h"
#include "PrimitiveSceneProxy.h"
#include "ScenePrivate.h"
#include "RenderUtils.h"
#include "UnifiedBuffer.h"
#include "CurveSkinningDefinitions.h"
#include "CurveSkinningSceneExtensionProxy.h"
#include "CurveSkinning/CurveSimulationProvider.h"
#include "ShaderPrintParameters.h"
#include "ShaderPrint.h"
#include "RenderGraphUtils.h"
#include "SceneRendering.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "ShaderCompilerCore.h"

DECLARE_STATS_GROUP(TEXT("CurveSkinningSceneExtension"), STATGROUP_CurveSkinningSceneExtension, STATCAT_Advanced);

DECLARE_DWORD_COUNTER_STAT(TEXT("Num Primitives"), STAT_CurveSkinningSceneExtension_NumPrimitives, STATGROUP_CurveSkinningSceneExtension);
DECLARE_DWORD_COUNTER_STAT(TEXT("Num Allocations"), STAT_CurveSkinningSceneExtension_NumAllocations, STATGROUP_CurveSkinningSceneExtension);
DECLARE_DWORD_COUNTER_STAT(TEXT("Num Updates"), STAT_CurveSkinningSceneExtension_NumUpdates, STATGROUP_CurveSkinningSceneExtension);

#if STATS
#define DECLARE_CURVESKINNING_MEMORY_STAT(CounterName, StatId, GroupId) \
	DECLARE_STAT(CounterName, StatId, GroupId, EStatDataType::ST_int64, EStatFlags::ClearEveryFrame, FPlatformMemory::MCR_Physical); \
	static DEFINE_STAT(StatId)
#else
#define DECLARE_CURVESKINNING_MEMORY_STAT(CounterName, StatId, GroupId)
#endif

DECLARE_CURVESKINNING_MEMORY_STAT(TEXT("Header Buffer Size"), STAT_CurveSkinningSceneExtension_HeaderBufferSize, STATGROUP_CurveSkinningSceneExtension);
DECLARE_CURVESKINNING_MEMORY_STAT(TEXT("Rest Position Buffer Size"), STAT_CurveSkinningSceneExtension_RestPositionBufferSize, STATGROUP_CurveSkinningSceneExtension);
DECLARE_CURVESKINNING_MEMORY_STAT(TEXT("Deformed Position Buffer Size"), STAT_CurveSkinningSceneExtension_DeformedPositionBufferSize, STATGROUP_CurveSkinningSceneExtension);

static int32 GCurveSkinningBuffersMinSizeBytes = 4 * 1024;
static FAutoConsoleVariableRef CVarCurveSkinningBuffersMinSizeBytes(
	TEXT("r.CurveSkinning.Buffers.MinSizeBytes"),
	GCurveSkinningBuffersMinSizeBytes,
	TEXT("The smallest size (in bytes) of the curve skinning data buffers."),
	ECVF_ReadOnly | ECVF_RenderThreadSafe
);

static int32 GCurveSkinningBuffersHeaderDataMinSizeBytes = 4 * 1024;
static FAutoConsoleVariableRef CVarCurveSkinningBuffersHeaderDataMinSizeBytes(
	TEXT("r.CurveSkinning.Buffers.HeaderDataMinSizeBytes"),
	GCurveSkinningBuffersHeaderDataMinSizeBytes,
	TEXT("The smallest size (in bytes) of the per-primitive curve skinning header data buffer."),
	ECVF_ReadOnly | ECVF_RenderThreadSafe
);

static bool GCurveSkinningBuffersDefrag = true;
static FAutoConsoleVariableRef CVarCurveSkinningBuffersDefrag(
	TEXT("r.CurveSkinning.Buffers.Defrag"),
	GCurveSkinningBuffersDefrag,
	TEXT("Whether or not to allow defragmentation of the curve skinning buffers."),
	ECVF_RenderThreadSafe
);

static float GCurveSkinningBuffersDefragLowWatermark = 0.375f;
static FAutoConsoleVariableRef CVarCurveSkinningBuffersDefragLowWatermark(
	TEXT("r.CurveSkinning.Buffers.Defrag.LowWatermark"),
	GCurveSkinningBuffersDefragLowWatermark,
	TEXT("Ratio of used to allocated memory at which to decide to defrag the curve skinning buffers."),
	ECVF_RenderThreadSafe
);

static int32 GCurveSkinningBuffersForceDefrag = 0;
static FAutoConsoleVariableRef CVarCurveSkinningBuffersForceDefrag(
	TEXT("r.CurveSkinning.Buffers.Defrag.Force"),
	GCurveSkinningBuffersForceDefrag,
	TEXT("0: Do not force defrag.\n")
	TEXT("1: Force one full defrag on the next update.\n")
	TEXT("2: Force a full defrag every frame."),
	ECVF_RenderThreadSafe
);

static int32 GCurveSkinningSimulationDebugDraw = 0;
static FAutoConsoleVariableRef CVarCurveSkinningSimulationDebugDraw(
	TEXT("r.CurveSkinning.Simulation.DebugDraw"),
	GCurveSkinningSimulationDebugDraw,
	TEXT("Enable curve skinning simulation debug visualisation.\n")
	TEXT("0: off (default)\n")
	TEXT("1: draw guide curves with per-instance hue + stats block"),
	ECVF_RenderThreadSafe
);

static int32 GCurveSkinningBuffersForceFullUpload = 0;
static FAutoConsoleVariableRef CVarCurveSkinningBuffersForceFullUpload(
	TEXT("r.CurveSkinning.Buffers.ForceFullUpload"),
	GCurveSkinningBuffersForceFullUpload,
	TEXT("0: Do not force a full upload.\n")
	TEXT("1: Force one full upload on the next update.\n")
	TEXT("2: Force a full upload every frame."),
	ECVF_RenderThreadSafe
);

// Scene uniform buffer declaration
BEGIN_SHADER_PARAMETER_STRUCT(FCurveSkinningSceneParameters, RENDERER_API)
	SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, Headers)
	SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, RestPositions)
	SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, DeformedPositions)
END_SHADER_PARAMETER_STRUCT()

DECLARE_SCENE_UB_STRUCT(FCurveSkinningSceneParameters, CurveSkinning, RENDERER_API)

static void GetDefaultCurveSkinningParameters(FCurveSkinningSceneParameters& OutParameters, FRDGBuilder& GraphBuilder)
{
	auto DefaultBuffer = GraphBuilder.CreateSRV(GSystemTextures.GetDefaultByteAddressBuffer(GraphBuilder, 4u));
	OutParameters.Headers          = DefaultBuffer;
	OutParameters.RestPositions    = DefaultBuffer;
	OutParameters.DeformedPositions = DefaultBuffer;
}
IMPLEMENT_SCENE_UB_STRUCT(FCurveSkinningSceneParameters, CurveSkinning, GetDefaultCurveSkinningParameters);

IMPLEMENT_SCENE_EXTENSION(FCurveSkinningSceneExtension);

bool FCurveSkinningSceneExtension::ShouldCreateExtension(FScene& InScene)
{
	return NaniteCurvesSupported() && DoesRuntimeSupportNanite(GetFeatureLevelShaderPlatform(InScene.GetFeatureLevel()), true, true);
}

FCurveSkinningSceneExtension::FCurveSkinningSceneExtension(FScene& InScene)
	: ISceneExtension(InScene)
{
}

FCurveSkinningSceneExtension::~FCurveSkinningSceneExtension()
{
}

void FCurveSkinningSceneExtension::InitExtension(FScene& InScene)
{
	const bool bNaniteEnabled = UseNanite(GetFeatureLevelShaderPlatform(InScene.GetFeatureLevel()));
	SetEnabled(NaniteCurvesSupported() && bNaniteEnabled);

	if (auto DataProvider = InScene.GetExtensionPtr<FCurveSkinningDataProvider>())
	{
		// Register default dummy wave-motion deformer
		DataProvider->RegisterProvider(
			GetCurveSimulationProviderId(),
			FCurveSkinningDataProvider::FOnProvideCurveData::CreateStatic(&ProvideCurveSimulationData)
		);
	}
}

ISceneExtensionUpdater* FCurveSkinningSceneExtension::CreateUpdater()
{
	return new FUpdater(*this);
}

ISceneExtensionRenderer* FCurveSkinningSceneExtension::CreateRenderer(FSceneRendererBase& InSceneRenderer, const FEngineShowFlags& EngineShowFlags)
{
	if (!IsEnabled() || !InSceneRenderer.GetViewFamily())
	{
		return nullptr;
	}

	return new FRenderer(InSceneRenderer, *this);
}

void FCurveSkinningSceneExtension::SetEnabled(bool bEnabled)
{
	if (bEnabled != IsEnabled())
	{
		if (bEnabled)
		{
			Buffers = MakeUnique<FBuffers>();
		}
		else
		{
			Buffers = nullptr;
			RestAllocator.Reset();
			DeformedAllocator.Reset();
			HeaderDatas.Reset();
			AllocatedHeaderDataIndices.Reset();
			SharedRestDatas.Reset();
			PendingRegistrations.Reset();
			PendingFreeIndices.Reset();
			HandleSlots.Reset();
			FreeHandleSlotIndices.Reset();
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// ICurveSkinningSceneExtension — public push-registration interface

ICurveSkinningSceneExtension* ICurveSkinningSceneExtension::Get(FSceneInterface& Scene)
{
	return static_cast<FScene&>(Scene).GetExtensionPtr<FCurveSkinningSceneExtension>();
}

FCurveSkinningHandle FCurveSkinningSceneExtension::Register(
	FPrimitiveSceneProxy* OwningProxy,
	TUniquePtr<FCurveSkinningSceneExtensionProxy>&& InData)
{
	// Register curve-skinning data for a primitive proxy. The extension takes
	// ownership of InData and stashes it as "pending" until the matching
	// FPrimitiveSceneInfo shows up in the next PostSceneUpdate change set
	// (at which point PersistentIndex / InstanceSceneDataOffset become valid
	// and the extension finalizes the header).

	// Serialized with Unregister and Pre/PostSceneUpdate by the RT command queue.
	check(IsInRenderingThread());
	check(OwningProxy != nullptr);
	check(InData.IsValid());

	if (!IsEnabled())
	{
		// Drop silently — caller will get an invalid handle and Unregister() will no-op.
		return FCurveSkinningHandle{};
	}

	// Allocate a handle slot. Reuse an inactive slot from the free-list when possible so
	// the slot's Generation counter survives across reuses and stale handles can be
	// distinguished from current ones.
	int32 SlotIndex;
	if (FreeHandleSlotIndices.Num() > 0)
	{
		SlotIndex = FreeHandleSlotIndices.Pop(EAllowShrinking::No);
	}
	else
	{
		SlotIndex = HandleSlots.AddDefaulted();
	}
	FHandleSlot& Slot = HandleSlots[SlotIndex];
	check(!Slot.bInUse);                 // free-list invariant: only inactive slots get recycled
	Slot.Proxy           = OwningProxy;
	Slot.PersistentIndex = INDEX_NONE;   // still pending — finalized in PostSceneUpdate
	Slot.Generation     += 1;            // monotonic across reuses (preserved by the free-list)
	Slot.bInUse          = true;

	FCurveSkinningHandle Handle;
	Handle.Index      = SlotIndex;
	Handle.Generation = Slot.Generation;

	FPendingRegistration Pending;
	Pending.Data   = MoveTemp(InData);
	Pending.Handle = Handle;
	PendingRegistrations.Emplace(OwningProxy, MoveTemp(Pending));

	return Handle;
}

void FCurveSkinningSceneExtension::Unregister(FCurveSkinningHandle Handle)
{
	// Release a previously-registered entry. Safe to call with an invalid
	// handle (no-op). If the entry was still pending (never finalized),
	// its data is freed immediately; otherwise the allocated GPU slot is
	// queued for release in the next PreSceneUpdate.
		
	check(IsInRenderingThread());

	if (!Handle.IsValid() || !HandleSlots.IsValidIndex(Handle.Index))
	{
		return;
	}

	FHandleSlot& Slot = HandleSlots[Handle.Index];
	if (!Slot.bInUse || Slot.Generation != Handle.Generation)
	{
		// Stale handle (slot was already freed and possibly recycled). Ignore.
		return;
	}

	if (Slot.PersistentIndex == INDEX_NONE)
	{
		// Still pending — finalize never ran. Drop the pending entry; the TUniquePtr
		// destructor frees the FCurveSkinningSceneExtensionProxy.
		PendingRegistrations.Remove(Slot.Proxy);
	}
	else
	{
		// Already finalized. Queue the persistent index for FreeBufferSpaceTask to
		// pick up in the next PreSceneUpdate.
		PendingFreeIndices.Add(Slot.PersistentIndex);
	}

	// Mark the slot inactive and push its index onto the free-list. We deliberately do
	// NOT destroy the slot (no RemoveAt) — its Generation must persist so that when a
	// future Register reuses this index it can bump Generation and invalidate any
	// surviving handle that still references the slot in its previous incarnation.
	Slot.bInUse          = false;
	Slot.Proxy           = nullptr;
	Slot.PersistentIndex = INDEX_NONE;
	FreeHandleSlotIndices.Add(Handle.Index);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// FBuffers

FCurveSkinningSceneExtension::FBuffers::FBuffers()
	: HeaderDataBuffer(GCurveSkinningBuffersHeaderDataMinSizeBytes >> 2u, TEXT("CurveSkinning.HeaderData"))
	, RestPositionBuffer(GCurveSkinningBuffersMinSizeBytes >> 2u, TEXT("CurveSkinning.RestPositions"))
	, DeformedPositionBuffer(GCurveSkinningBuffersMinSizeBytes >> 2u, TEXT("CurveSkinning.DeformedPositions"))
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Buffer upload

void FCurveSkinningSceneExtension::FinishCurveSkinningBufferUpload(
	FRDGBuilder& GraphBuilder,
	FCurveSkinningSceneParameters* OutParams,
	bool bUpdateStats)
{
	if (!IsEnabled())
	{
		return;
	}

	// Wait for all three tasks to complete before touching the uploaders on the render thread.
	// AllocBufferSpaceTask accumulates all scatter uploader Add() calls, so it must finish
	// before ResizeAndUploadTo is called below.
	UE::Tasks::Wait(
		MakeArrayView(
			{
				TaskHandles[FreeBufferSpaceTask],
				TaskHandles[InitHeaderDataTask],
				TaskHandles[AllocBufferSpaceTask]
			}
		)
	);

	const uint32 MinHeaderDataSize = (HeaderDatas.GetMaxIndex() + 1);
	const uint32 MinRestDataSize = RestAllocator.GetMaxSize();
	const uint32 MinDeformedDataSize = DeformedAllocator.GetMaxSize();

	FRDGBufferRef HeaderBuffer = nullptr;
	FRDGBufferRef RestBuffer = nullptr;
	FRDGBufferRef DeformedBuffer = nullptr;

	RDG_GPU_MASK_SCOPE(GraphBuilder, FRHIGPUMask::All());

	if (Uploader.IsValid())
	{
		HeaderBuffer = Uploader->HeaderDataUploader.ResizeAndUploadTo(
			GraphBuilder,
			Buffers->HeaderDataBuffer,
			MinHeaderDataSize
		);

		RestBuffer = Uploader->RestPositionUploader.ResizeAndUploadTo(
			GraphBuilder,
			Buffers->RestPositionBuffer,
			MinRestDataSize
		);

		DeformedBuffer = Uploader->DeformedPositionUploader.ResizeAndUploadTo(
			GraphBuilder,
			Buffers->DeformedPositionBuffer,
			MinDeformedDataSize
		);

		Uploader = nullptr;
	}
	else
	{
		HeaderBuffer   = Buffers->HeaderDataBuffer.ResizeBufferIfNeeded(GraphBuilder, MinHeaderDataSize);
		RestBuffer     = Buffers->RestPositionBuffer.ResizeBufferIfNeeded(GraphBuilder, MinRestDataSize);
		DeformedBuffer = Buffers->DeformedPositionBuffer.ResizeBufferIfNeeded(GraphBuilder, MinDeformedDataSize);
	}

	if (bUpdateStats)
	{
		INC_DWORD_STAT_BY(STAT_CurveSkinningSceneExtension_NumPrimitives, HeaderDatas.Num());
		INC_MEMORY_STAT_BY(STAT_CurveSkinningSceneExtension_HeaderBufferSize, HeaderBuffer->Desc.GetSize());
		INC_MEMORY_STAT_BY(STAT_CurveSkinningSceneExtension_RestPositionBufferSize, RestBuffer->Desc.GetSize());
		INC_MEMORY_STAT_BY(STAT_CurveSkinningSceneExtension_DeformedPositionBufferSize, DeformedBuffer->Desc.GetSize());
	}

	if (OutParams != nullptr)
	{
		OutParams->Headers          = GraphBuilder.CreateSRV(HeaderBuffer);
		OutParams->RestPositions    = GraphBuilder.CreateSRV(RestBuffer);
		OutParams->DeformedPositions= GraphBuilder.CreateSRV(DeformedBuffer);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Perform curve skinning (broadcast to providers)

void FCurveSkinningSceneExtension::PerformCurveSkinning(
	FCurveSkinningSceneParameters& Parameters,
	FRDGBuilder& GraphBuilder,
	const FGameTime& CurrentTime)
{
	RDG_EVENT_SCOPE(GraphBuilder, "CurveSkinning");

	auto DataProvider = Scene.GetExtensionPtr<FCurveSkinningDataProvider>();
	if (!DataProvider || !DataProvider->HasProviders())
	{
		return;
	}

	if (HeaderDatas.Num() == 0 || AllocatedHeaderDataIndices.Num() == 0)
	{
		return;
	}

	const TArray<FGuid> ProviderIds = DataProvider->GetProviderIds();

	TArray<FCurveSkinningDataProvider::FProviderRange, TInlineAllocator<8>> ProviderRanges;
	ProviderRanges.Reserve(ProviderIds.Num());
	for (const FGuid& ProviderId : ProviderIds)
	{
		FCurveSkinningDataProvider::FProviderRange& Range = ProviderRanges.Emplace_GetRef();
		Range.Id = ProviderId;
		Range.Count = 0;
		Range.Offset = 0;
	}

	TArray<uint8, FConcurrentLinearArrayAllocator> PrimitivesToRangeIndex;
	PrimitivesToRangeIndex.AddUninitialized(AllocatedHeaderDataIndices.Num());

	TArrayView<FPrimitiveSceneInfo*> Primitives = GraphBuilder.AllocPODArrayView<FPrimitiveSceneInfo*>(AllocatedHeaderDataIndices.Num());
	TArrayView<FCurveSkinningSceneExtensionProxy*> Proxies = GraphBuilder.AllocPODArrayView<FCurveSkinningSceneExtensionProxy*>(AllocatedHeaderDataIndices.Num());

	// Scratch — per-primitive header index + dirty mask, to be re-bucketed per provider below.
	// We no longer pre-compute byte offsets here; the shader reads everything from the
	// curve-skinning header buffer via LoadCurveSkinningHeader(HeaderIndex).
	struct FScratch
	{
		uint32          HeaderIndex;
		uint32          NumCurves;
		EDirtyCurveData DirtyCurveData;
	};
	TArrayView<FScratch> Scratch = GraphBuilder.AllocPODArrayView<FScratch>(AllocatedHeaderDataIndices.Num());

	uint32 PrimitiveCount = 0;
	for (const int32 HeaderDataIndex : AllocatedHeaderDataIndices)
	{
		FHeaderData& Header = HeaderDatas[HeaderDataIndex];
		Header.Validate();

		if (!EnumHasAnyFlags(Header.DirtyCurveData, EDirtyCurveData::All))
		{
			continue;
		}

		int32 RangeIndex = 0;
		for (; RangeIndex < ProviderRanges.Num(); ++RangeIndex)
		{
			if (Header.ProviderId == ProviderRanges[RangeIndex].Id)
			{
				++ProviderRanges[RangeIndex].Count;
				break;
			}
		}

		check(RangeIndex != ProviderRanges.Num());

		PrimitivesToRangeIndex[PrimitiveCount] = (uint8)RangeIndex;
		Primitives[PrimitiveCount]             = Header.PrimitiveSceneInfo;
		Proxies[PrimitiveCount]                = Header.Proxy.Get();
		Scratch[PrimitiveCount].HeaderIndex    = (uint32)HeaderDataIndex;
		Scratch[PrimitiveCount].NumCurves      = Header.NumCurves;
		Scratch[PrimitiveCount].DirtyCurveData = Header.DirtyCurveData;

		Header.DirtyCurveData = EDirtyCurveData::None;

		++PrimitiveCount;
	}

	uint32 IndirectionCount = 0;
	for (FCurveSkinningDataProvider::FProviderRange& Range : ProviderRanges)
	{
		Range.Offset = IndirectionCount;
		IndirectionCount += Range.Count;
		Range.Count = 0;
	}

	TArrayView<FCurveSkinningDataProvider::FProviderIndirection> PrimitiveIndices =
		GraphBuilder.AllocPODArrayView<FCurveSkinningDataProvider::FProviderIndirection>(IndirectionCount);

	for (uint32 PrimitiveIndex = 0; PrimitiveIndex < PrimitiveCount; ++PrimitiveIndex)
	{
		FCurveSkinningDataProvider::FProviderRange& Range = ProviderRanges[PrimitivesToRangeIndex[PrimitiveIndex]];
		PrimitiveIndices[Range.Offset + Range.Count] = FCurveSkinningDataProvider::FProviderIndirection(
			Scratch[PrimitiveIndex].HeaderIndex,
			Scratch[PrimitiveIndex].NumCurves,
			Scratch[PrimitiveIndex].DirtyCurveData
		);
		++Range.Count;
	}

	// Hand providers direct refs to the RDG-registered curve-skinning buffers.
	// FinishCurveSkinningBufferUpload() ran in FRenderer::UpdateSceneUniformBuffer just
	// before us, so Parameters.{Headers,RestPositions,DeformedPositions} are valid
	// SRVs on this graph builder. Providers bind their own compute passes against
	// these buffers and need no per-frame scatter upload — per-instance state is read
	// from the header buffer (and the GPU scene via Header.InstanceSceneDataOffset)
	// inside the shader.
	FCurveSkinningDataProvider::FProviderContext Context(
		Primitives.Left(PrimitiveCount),
		Proxies.Left(PrimitiveCount),
		PrimitiveIndices,
		CurrentTime,
		GraphBuilder,
		Parameters.Headers->GetParent(),
		Parameters.RestPositions->GetParent(),
		Parameters.DeformedPositions->GetParent()
	);

	DataProvider->Broadcast(ProviderRanges, Context);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Defragmentation

bool FCurveSkinningSceneExtension::ProcessBufferDefragmentation()
{
	RestAllocator.Consolidate();
	DeformedAllocator.Consolidate();

	const bool bAllowDefrag = GCurveSkinningBuffersDefrag;
	static const int32 MinDeformedBufferCount = GCurveSkinningBuffersMinSizeBytes / sizeof(float);
	const float LowWaterMarkRatio = GCurveSkinningBuffersDefragLowWatermark;
	const int32 EffectiveMaxSize = FMath::RoundUpToPowerOfTwo(DeformedAllocator.GetMaxSize());
	const int32 LowWaterMark = uint32(EffectiveMaxSize * LowWaterMarkRatio);
	const int32 UsedSize = DeformedAllocator.GetSparselyAllocatedSize();

	if (!bAllowDefrag)
	{
		return false;
	}

	const bool bForceDefrag = GCurveSkinningBuffersForceDefrag != 0;
	if (GCurveSkinningBuffersForceDefrag == 1)
	{
		GCurveSkinningBuffersForceDefrag = 0;
	}

	if (!bForceDefrag && (EffectiveMaxSize <= MinDeformedBufferCount || UsedSize > LowWaterMark))
	{
		return false;
	}

	RestAllocator.Reset();
	DeformedAllocator.Reset();
	AllocatedHeaderDataIndices.Reset();
	SharedRestDatas.Reset();

	for (auto& Data : HeaderDatas)
	{
		if (Data.RestBufferOffset != INDEX_NONE)
		{
			Data.RestBufferOffset = INDEX_NONE;
			Data.RestBufferCount = 0;
		}

		if (Data.DeformedBufferOffset != INDEX_NONE)
		{
			Data.DeformedBufferOffset = INDEX_NONE;
			Data.DeformedBufferCount = 0;
		}

		Data.bIsAllocated = false;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Allocation helpers

void FCurveSkinningSceneExtension::AllocSpaceForPrimitive(FHeaderData& HeaderData, FUploader& InUploader)
{
	const uint32 TotalPoints = (uint32)HeaderData.NumCurves * (uint32)HeaderData.NumPointsPerCurve;
	const uint32 RestFloatCount = TotalPoints * CURVESKINNING_FLOATS_PER_POINT;
	const uint32 DeformedFloatCount = TotalPoints * CURVESKINNING_FLOATS_PER_POINT * 2; // x2 for double buffering

	// Share rest data across instances of the same source asset.
	// The proxy carries an opaque key set by the caller (e.g. the UGroomAsset pointer).
	// The extension never dereferences this key — it is used purely for identity.
	FRestDataKey RestKey;
	RestKey.AssetPtr = HeaderData.Proxy->GetRestDataKey();

	FSharedRestData* SharedRest = SharedRestDatas.Find(RestKey);
	if (SharedRest)
	{
		SharedRest->RefCount++;
		HeaderData.RestBufferOffset = SharedRest->RestBufferOffset;
		HeaderData.RestBufferCount = SharedRest->RestBufferCount;
	}
	else
	{
		HeaderData.RestBufferCount = RestFloatCount;
		HeaderData.RestBufferOffset = RestAllocator.Allocate(RestFloatCount);

		FSharedRestData NewShared;
		NewShared.RestBufferOffset = HeaderData.RestBufferOffset;
		NewShared.RestBufferCount = HeaderData.RestBufferCount;
		NewShared.RefCount = 1;
		SharedRestDatas.Add(RestKey, NewShared);

		// Upload rest positions from the proxy's CPU-side data (populated at proxy construction
		// from the asset's pre-built FCurveSkinningData). Rest data is fully valid here —
		// no deferred GPU initialisation needed.
		TConstArrayView<float> RestPositions = HeaderData.Proxy->GetRestPositions();
		for (int32 FloatIdx = 0; FloatIdx < RestPositions.Num(); ++FloatIdx)
		{
			InUploader.RestPositionUploader.Add(RestPositions[FloatIdx], HeaderData.RestBufferOffset + FloatIdx);
		}
	}

	// Allocate deformed buffer space (per-instance, double-buffered)
	HeaderData.DeformedBufferCount = DeformedFloatCount;
	HeaderData.DeformedBufferOffset = DeformedAllocator.Allocate(DeformedFloatCount);

	HeaderData.bIsAllocated = true;
	HeaderData.DirtyCurveData = EDirtyCurveData::All;

	// Initialise both deformed buffer slots to the rest positions so the first frame
	// before any deformer dispatch renders the curves correctly in rest pose.
	// This also covers the post-defragmentation case since AllocSpaceForPrimitive
	// is called again for every primitive after a defrag reset.
	{
		TConstArrayView<float> RestPositions = HeaderData.Proxy->GetRestPositions();
		for (int32 FloatIdx = 0; FloatIdx < (int32)RestFloatCount; ++FloatIdx)
		{
			const float Value = RestPositions.IsValidIndex(FloatIdx) ? RestPositions[FloatIdx] : 0.0f;
			// Slot 0 — current deformed slot
			InUploader.DeformedPositionUploader.Add(Value, HeaderData.DeformedBufferOffset + (uint32)FloatIdx);
			// Slot 1 — previous deformed slot (double-buffer ping-pong partner)
			InUploader.DeformedPositionUploader.Add(Value, HeaderData.DeformedBufferOffset + RestFloatCount + (uint32)FloatIdx);
		}
	}
}

void FCurveSkinningSceneExtension::FreeSpaceForPrimitive(FHeaderData& HeaderData)
{
	if (HeaderData.RestBufferOffset != INDEX_NONE)
	{
		FRestDataKey RestKey;
		RestKey.AssetPtr = HeaderData.Proxy->GetRestDataKey();

		if (FSharedRestData* SharedRest = SharedRestDatas.Find(RestKey))
		{
			SharedRest->RefCount--;
			if (SharedRest->RefCount <= 0)
			{
				RestAllocator.Free(SharedRest->RestBufferOffset, SharedRest->RestBufferCount);
				SharedRestDatas.Remove(RestKey);
			}
		}

		HeaderData.RestBufferOffset = INDEX_NONE;
		HeaderData.RestBufferCount = 0;
	}

	if (HeaderData.DeformedBufferOffset != INDEX_NONE)
	{
		DeformedAllocator.Free(HeaderData.DeformedBufferOffset, HeaderData.DeformedBufferCount);
		HeaderData.DeformedBufferOffset = INDEX_NONE;
		HeaderData.DeformedBufferCount = 0;
	}

	HeaderData.bIsAllocated = false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// FUpdater

FCurveSkinningSceneExtension::FUpdater::FUpdater(FCurveSkinningSceneExtension& InSceneData)
	: SceneData(&InSceneData)
{
	SceneData->UpdateCounter++;
}

void FCurveSkinningSceneExtension::FUpdater::End()
{
	UE::Tasks::Wait(SceneData->TaskHandles);
}

void FCurveSkinningSceneExtension::FUpdater::PreSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePreUpdateChangeSet& ChangeSet)
{
	if (!SceneData->IsEnabled())
	{
		return;
	}

	TRACE_CPUPROFILER_EVENT_SCOPE(FCurveSkinningSceneExtension::FUpdater::PreSceneUpdate);

	// Drain Unregister-queued frees on the render thread, BEFORE launching the async
	// task. The task captures the local copy by move, so any RegisterPending/Unregister
	// calls that fire on the RT after this point land in a fresh empty list and are
	// processed next frame. See ICurveSkinningSceneExtension threading model.
	TArray<int32> PendingFreesSnapshot = MoveTemp(SceneData->PendingFreeIndices);

	SceneData->TaskHandles[FreeBufferSpaceTask] = GraphBuilder.AddSetupTask(
		[this, PendingFrees = MoveTemp(PendingFreesSnapshot)]
		{
			TRACE_CPUPROFILER_EVENT_SCOPE(CurveSkinning::FreeBufferSpace);

			// DRTR is the only path that puts entries here, and it runs on the RT during
			// the remove phase of FScene::Update — strictly before PreSceneUpdate. Every
			// removed groom is already in PendingFrees by the time this task launches; we
			// deliberately do not walk ChangeSet.RemovedPrimitiveIds (which can be in the
			// millions during streaming-out events).
			for (int32 PersistentIndex : PendingFrees)
			{
				if (SceneData->HeaderDatas.IsValidIndex(PersistentIndex))
				{
					FHeaderData& Data = SceneData->HeaderDatas[PersistentIndex];

					if (Data.bIsAllocated)
					{
						SceneData->FreeSpaceForPrimitive(Data);
						SceneData->AllocatedHeaderDataIndices.Remove(PersistentIndex);
					}

					SceneData->HeaderDatas.RemoveAt(PersistentIndex);
				}
			}

			bForceFullUpload = GCurveSkinningBuffersForceFullUpload != 0;
			if (GCurveSkinningBuffersForceFullUpload == 1)
			{
				GCurveSkinningBuffersForceFullUpload = 0;
			}

			const bool bDefragging = SceneData->ProcessBufferDefragmentation();
			bForceFullUpload |= bDefragging;
		},
		UE::Tasks::ETaskPriority::Normal,
		true /* bAsync */
	);
}

void FCurveSkinningSceneExtension::FUpdater::PostSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePostUpdateChangeSet& ChangeSet)
{
	if (!SceneData->IsEnabled())
	{
		return;
	}

	TRACE_CPUPROFILER_EVENT_SCOPE(FCurveSkinningSceneExtension::FUpdater::PostSceneUpdate);

	// Drain Register-queued entries on the render thread, BEFORE launching the async
	// task. Any Register/Unregister calls that fire on the RT after this point land in
	// a fresh empty map and are picked up on the next PostSceneUpdate.
	TMap<FPrimitiveSceneProxy*, FPendingRegistration> PendingSnapshot = MoveTemp(SceneData->PendingRegistrations);

	if (PendingSnapshot.Num() > 0)
	{
		SceneData->TaskHandles[InitHeaderDataTask] = GraphBuilder.AddSetupTask(
			[this, Pending = MoveTemp(PendingSnapshot)]() mutable
			{
				TRACE_CPUPROFILER_EVENT_SCOPE(CurveSkinning::InitHeaderData);

				// Drive the join from Pending (small — bounded by the number of grooms
				// registered this frame) rather than from ChangeSet.AddedPrimitiveSceneInfos
				// (which can be in the millions during streaming-in). The proxy's
				// PrimitiveSceneInfo back-pointer is set by FPrimitiveSceneInfo's ctor
				// before CRTR runs, and CRTR is what called Register() to populate Pending,
				// so by here the back-pointer + PersistentIndex are guaranteed valid.
				for (TPair<FPrimitiveSceneProxy*, FPendingRegistration>& Entry : Pending)
				{
					FPrimitiveSceneProxy* Proxy = Entry.Key;
					FPendingRegistration& PendingEntry = Entry.Value;

					FPrimitiveSceneInfo* PrimitiveSceneInfo = Proxy ? Proxy->GetPrimitiveSceneInfo() : nullptr;
					if (!ensureMsgf(PrimitiveSceneInfo != nullptr, TEXT("FCurveSkinningSceneExtension: pending registration has no PrimitiveSceneInfo back-pointer; CRTR/AddPrimitiveSceneInfo_RT ordering may have changed.")))
					{
						continue;
					}

					const int32 PersistentIndex = PrimitiveSceneInfo->GetPersistentIndex().Index;
					if (!ensureMsgf(PersistentIndex != INDEX_NONE, TEXT("FCurveSkinningSceneExtension: pending registration's PSI has no PersistentIndex.")))
					{
						continue;
					}

					check(PendingEntry.Data.IsValid());

					FHeaderData NewHeader;
					NewHeader.InstanceSceneDataOffset     = PrimitiveSceneInfo->GetInstanceSceneDataOffset();
					NewHeader.NumInstanceSceneDataEntries = PrimitiveSceneInfo->GetNumInstanceSceneDataEntries();
					NewHeader.ProviderId                  = PendingEntry.Data->GetDataProviderId();
					NewHeader.PrimitiveSceneInfo          = PrimitiveSceneInfo;
					NewHeader.NumCurves                   = (uint16)PendingEntry.Data->GetNumCurves();
					NewHeader.NumPointsPerCurve           = (uint8)PendingEntry.Data->GetNumPointsPerCurve();
					NewHeader.Proxy                       = MoveTemp(PendingEntry.Data);

					SceneData->HeaderDatas.EmplaceAt(PersistentIndex, MoveTemp(NewHeader));

					// Record the persistent index against the originating handle so a
					// later Unregister() routes through the finalized free path.
					if (SceneData->HandleSlots.IsValidIndex(PendingEntry.Handle.Index))
					{
						FHandleSlot& Slot = SceneData->HandleSlots[PendingEntry.Handle.Index];
						if (Slot.bInUse && Slot.Generation == PendingEntry.Handle.Generation)
						{
							Slot.PersistentIndex = PersistentIndex;
						}
					}

					AllocationUpdateList.Emplace(PersistentIndex);
				}
			},
			SceneData->TaskHandles[FreeBufferSpaceTask],
			UE::Tasks::ETaskPriority::Normal,
			true /* bAsync */
		);
	}
	// Allocate buffer space for new primitives
	SceneData->TaskHandles[AllocBufferSpaceTask] = GraphBuilder.AddSetupTask(
		[this]
		{
			TRACE_CPUPROFILER_EVENT_SCOPE(CurveSkinning::AllocBufferSpace);

			// Need an uploader for any of: scene-update alloc changes, forced full upload, or
			// the per-frame slot refresh (any allocated primitive — see the refresh loop below).
			if (!SceneData->Uploader.IsValid() && (AllocationUpdateList.Num() > 0 || bForceFullUpload || SceneData->AllocatedHeaderDataIndices.Num() > 0))
			{
				SceneData->Uploader = MakeUnique<FUploader>();
			}

			if (!SceneData->Uploader.IsValid())
			{
				return;
			}

			FUploader& CurrentUploader = *SceneData->Uploader;

			if (bForceFullUpload)
			{
				for (typename TSparseArray<FHeaderData>::TIterator It(SceneData->HeaderDatas); It; ++It)
				{
					FHeaderData& HeaderData = *It;
					if (!HeaderData.bIsAllocated)
					{
						SceneData->AllocSpaceForPrimitive(HeaderData, CurrentUploader);
						SceneData->AllocatedHeaderDataIndices.Emplace(It.GetIndex());
					}

					CurrentUploader.HeaderDataUploader.Add(HeaderData.Pack(), It.GetIndex());
					HeaderDataUpdateList.Emplace(It.GetIndex());
				}
			}
			else
			{
				for (const int32 HeaderDataIndex : AllocationUpdateList)
				{
					if (SceneData->HeaderDatas.IsValidIndex(HeaderDataIndex))
					{
						FHeaderData& HeaderData = SceneData->HeaderDatas[HeaderDataIndex];

						if (HeaderData.bIsAllocated)
						{
							SceneData->FreeSpaceForPrimitive(HeaderData);
							SceneData->AllocatedHeaderDataIndices.Remove(HeaderDataIndex);
						}

						SceneData->AllocSpaceForPrimitive(HeaderData, CurrentUploader);
						SceneData->AllocatedHeaderDataIndices.Emplace(HeaderDataIndex);
					}
				}
			}

			// Per-frame slot refresh: re-upload Pack() for every allocated primitive so that
			// CurrentSlot flips done by AdvanceSimulationFrame become visible to shaders. Pack()
			// bakes the slot bit into DeformedBufferOffset/PrevDeformedBufferOffset, so without
			// this refresh the GPU header is frozen at allocation time and the velocity pass
			// reads the rest-pose instead of last frame's deformed positions.
			//
			// bForceFullUpload above already re-uploads every entry, so skip in that path.
			if (!bForceFullUpload)
			{
				for (const int32 HeaderDataIndex : SceneData->AllocatedHeaderDataIndices)
				{
					if (SceneData->HeaderDatas.IsValidIndex(HeaderDataIndex))
					{
						FHeaderData& HeaderData = SceneData->HeaderDatas[HeaderDataIndex];
						if (HeaderData.bIsAllocated)
						{
							CurrentUploader.HeaderDataUploader.Add(HeaderData.Pack(), HeaderDataIndex);
							HeaderDataUpdateList.Emplace(HeaderDataIndex);
						}
					}
				}
			}

			INC_DWORD_STAT_BY(STAT_CurveSkinningSceneExtension_NumAllocations, AllocationUpdateList.Num());
			INC_DWORD_STAT_BY(STAT_CurveSkinningSceneExtension_NumUpdates, HeaderDataUpdateList.Num());

			AllocationUpdateList.Reset();
			HeaderDataUpdateList.Reset();
		},
		MakeArrayView(
			{
				SceneData->TaskHandles[FreeBufferSpaceTask],
				SceneData->TaskHandles[InitHeaderDataTask]
			}
		),
		UE::Tasks::ETaskPriority::Normal,
		true /* bAsync */
	);
}

void FCurveSkinningSceneExtension::AdvanceSimulationFrame()
{
	// Pure CPU bookkeeping — no RDG work. Called once per render-thread frame from
	// FRenderer::UpdateSceneUniformBuffer right after PerformCurveSkinning, so the slot
	// flipped here defines next frame's "current" target (and this frame's output
	// automatically becomes next frame's "previous" via PrevDeformedBufferOffset).
	for (const int32 HeaderDataIndex : AllocatedHeaderDataIndices)
	{
		if (HeaderDatas.IsValidIndex(HeaderDataIndex))
		{
			FHeaderData& Header   = HeaderDatas[HeaderDataIndex];
			Header.DirtyCurveData |= EDirtyCurveData::Current;
			Header.CurrentSlot     = !Header.CurrentSlot;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Debug

class FGlobalCurveSolverDebugCS : public FGlobalShader
{
public:
	static constexpr uint32 CurvesPerGroup = 64u;

	DECLARE_GLOBAL_SHADER(FGlobalCurveSolverDebugCS);
	SHADER_USE_PARAMETER_STRUCT(FGlobalCurveSolverDebugCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)
		SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrint)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, InstanceList)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, UniqueGroomList)
		SHADER_PARAMETER(uint32, NumInstances)
		SHADER_PARAMETER(uint32, NumUniqueGrooms)
		SHADER_PARAMETER(uint32, TotalNumCurves)
		SHADER_PARAMETER(uint32, TotalNumPoints)
		END_SHADER_PARAMETER_STRUCT()

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShaderPrint::IsSupported(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.CompilerFlags.Add(CFLAG_HLSL2021);
		OutEnvironment.SetDefine(TEXT("CURVES_PER_GROUP"),                  CurvesPerGroup);
		OutEnvironment.SetDefine(TEXT("VF_SUPPORTS_PRIMITIVE_SCENE_DATA"),   1);
	}
};

IMPLEMENT_GLOBAL_SHADER(FGlobalCurveSolverDebugCS, "/Engine/Private/CurveSkinning/CurveSkinningDebug.usf", "CurveSkinningDebugCS", SF_Compute);

void FCurveSkinningSceneExtension::FRenderer::AddDebugPass(FRDGBuilder& GraphBuilder, const TArray<FViewInfo>& Views, FCurveSkinningSceneExtension* SceneData, FSceneUniformBuffer& SceneUniformBuffer)
{
	if (Views.IsEmpty())
	{
		return;
	}

	const FViewInfo& View = Views[0];
	if (!ShaderPrint::IsSupported(View.GetShaderPlatform()))
	{
		return;
	}

	ShaderPrint::SetEnabled(true);

	// Build per-instance list from extension's header data.
	// Key insight: AllocatedHeaderDataIndices stores FPersistentPrimitiveIndex::Index values
	// which are the same indices LoadCurveSkinningHeader() uses to address the GPU header buffer.
	// InstanceSceneDataOffset is used by the debug shader to fetch LocalToWorld for WS transform.
	struct FInstanceDesc { uint32 PrimIndex; uint32 UniqueGroomIndex; uint32 InstanceSceneDataOffset; uint32 Pad; };
	struct FUniqueGroomDesc { uint32 NumInstances; uint32 NumCurves; uint32 NumPointsPerCurve; uint32 Pad; };

	TArray<FInstanceDesc> InstanceList;
	TArray<FUniqueGroomDesc> UniqueGroomList;
	TMap<const void*, uint32> GroomKeyToIdx;

	uint32 TotalPoints  = 0;
	uint32 TotalCurves  = 0;

	for (const int32 HeaderIdx : SceneData->AllocatedHeaderDataIndices)
	{
		if (!SceneData->HeaderDatas.IsValidIndex(HeaderIdx))
		{
			continue;
		}
		const FHeaderData& Header = SceneData->HeaderDatas[HeaderIdx];
		if (!Header.bIsAllocated || !Header.Proxy)
		{
			continue;
		}

		const void* GroomKey = Header.Proxy->GetRestDataKey();
		const uint32* ExistingIdx = GroomKeyToIdx.Find(GroomKey);
		uint32 UniqueGroomIdx;
		if (ExistingIdx)
		{
			UniqueGroomIdx = *ExistingIdx;
			UniqueGroomList[UniqueGroomIdx].NumInstances++;
		}
		else
		{
			UniqueGroomIdx = UniqueGroomList.Num();
			GroomKeyToIdx.Add(GroomKey, UniqueGroomIdx);

			FUniqueGroomDesc& UniqueGroom = UniqueGroomList.AddDefaulted_GetRef();
			UniqueGroom.NumInstances = 1;
			UniqueGroom.NumCurves = Header.NumCurves;
			UniqueGroom.NumPointsPerCurve = Header.NumPointsPerCurve;
			UniqueGroom.Pad = 0;
		}

		FInstanceDesc& Desc             = InstanceList.AddDefaulted_GetRef();
		Desc.PrimIndex                  = (uint32)HeaderIdx;
		Desc.UniqueGroomIndex           = UniqueGroomIdx;
		Desc.InstanceSceneDataOffset    = Header.InstanceSceneDataOffset;
		Desc.Pad                        = 0;

		TotalCurves += Header.NumCurves;
		TotalPoints += (uint32)Header.NumCurves * Header.NumPointsPerCurve;
	}

	const uint32 NumInstances    = InstanceList.Num();
	const uint32 NumUniqueGrooms = UniqueGroomList.Num();

	if (NumInstances > 0)
	{
		FRDGBufferRef InstanceListBuffer = CreateUploadBuffer(
			GraphBuilder,
			TEXT("CurveSkinning.Debug.InstanceList"),
			sizeof(FInstanceDesc),
			FMath::Max(NumInstances, 1u),
			InstanceList.GetData(),
			NumInstances * sizeof(FInstanceDesc));

		FRDGBufferRef UniqueGroomListBuffer = CreateUploadBuffer(
			GraphBuilder,
			TEXT("CurveSkinning.Debug.GroomCounts"),
			sizeof(FUniqueGroomDesc),
			FMath::Max(NumUniqueGrooms, 1u),
			UniqueGroomList.GetData(),
			NumUniqueGrooms * sizeof(FUniqueGroomDesc));

		// GetBuffer() re-creates the RDG UB (bAnyMemberDirty is set after our Set() call).
		const TRDGUniformBufferRef<FSceneUniformParameters> SceneUBRef = SceneUniformBuffer.GetBuffer(GraphBuilder);

		{
			check(NumInstances > 0);
			check(InstanceListBuffer != nullptr);
			check(UniqueGroomListBuffer  != nullptr);

			// Reserve ShaderPrint line budget.	
			// x2: deformed lines + rest lines (user may enable both checkboxes).
			ShaderPrint::RequestSpaceForLines(TotalPoints * 2u + TotalCurves + 64u);
			ShaderPrint::RequestSpaceForCharacters(512u);

			auto* PassParameters = GraphBuilder.AllocParameters<FGlobalCurveSolverDebugCS::FParameters>();
			PassParameters->Scene               = SceneUBRef;
			PassParameters->InstanceList        = GraphBuilder.CreateSRV(InstanceListBuffer, PF_R32G32B32A32_UINT);
			PassParameters->UniqueGroomList     = GraphBuilder.CreateSRV(UniqueGroomListBuffer, PF_R32G32B32A32_UINT);
			PassParameters->NumInstances        = NumInstances;
			PassParameters->NumUniqueGrooms     = NumUniqueGrooms;
			PassParameters->TotalNumCurves		= TotalCurves;
			PassParameters->TotalNumPoints		= TotalPoints;
			ShaderPrint::SetParameters(GraphBuilder, View.ShaderPrintData, PassParameters->ShaderPrint);

			auto ComputeShader = View.ShaderMap->GetShader<FGlobalCurveSolverDebugCS>();

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("CurveSkinning::DebugDraw(Instances=%u Grooms=%u)", NumInstances, NumUniqueGrooms),
				ComputeShader,
				PassParameters,
				FIntVector(NumInstances, 1, 1));
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// FRenderer

void FCurveSkinningSceneExtension::FRenderer::UpdateSceneUniformBuffer(FRDGBuilder& GraphBuilder, FSceneUniformBuffer& SceneUniformBuffer)
{
	SCOPED_NAMED_EVENT(FCurveSkinningSceneExtension_FRenderer_UpdateSceneUniformBuffer, FColor::Silver);
	check(SceneData->IsEnabled());

	FCurveSkinningSceneParameters Parameters;
	const bool bUpdateStats = true;
	SceneData->FinishCurveSkinningBufferUpload(GraphBuilder, &Parameters, bUpdateStats);

	// Run the curve simulation at most once per render-thread frame, regardless of how
	// many renderers (split-screen, scene captures, planar reflections, etc.) bind the
	// scene UB. The first renderer dispatched wins; subsequent renderers in the same
	// frame see the same DeformedPositions and reuse them via the scene UB below.
	//
	// FinishCurveSkinningBufferUpload() above has waited on the
	// FreeBufferSpace / InitHeaderData / AllocBufferSpace tasks and registered the
	// buffers in RDG — so Parameters.{Headers,RestPositions,DeformedPositions} are
	// valid SRVs and the dispatch can land directly on this graph builder.
	if (SceneData->LastUpdateFrameNumber != GFrameNumberRenderThread)
	{
		// 1. Run skinning
		const FSceneViewFamily* ViewFamily = GetSceneRenderer().GetViewFamily();
		const FGameTime CurrentTime = ViewFamily ? ViewFamily->Time : FGameTime();
		SceneData->PerformCurveSkinning(Parameters, GraphBuilder, CurrentTime);

		// 2. Advance simulation
		// Advance simulation state for next frame (slot ping-pong + dirty re-arm).
		// Must happen after PerformCurveSkinning so this frame's sim wrote into the
		// slot we're about to demote to "previous". See AdvanceSimulationFrame() comment.
		SceneData->AdvanceSimulationFrame();

		SceneData->LastUpdateFrameNumber = GFrameNumberRenderThread;
	}

	SceneUniformBuffer.Set(SceneUB::CurveSkinning, Parameters);

	if (GCurveSkinningSimulationDebugDraw != 0)
	{
		AddDebugPass(GraphBuilder, GetSceneRenderer().Views, SceneData, SceneUniformBuffer);
	}
}