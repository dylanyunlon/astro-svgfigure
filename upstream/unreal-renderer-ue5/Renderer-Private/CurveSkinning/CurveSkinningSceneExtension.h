// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "SpanAllocator.h"
#include "SceneExtensions.h"
#include "CurveSkinningDefinitions.h"
#include "CurveSkinningSceneExtensionProxy.h"
#include "CurveSkinning/CurveSkinningSceneExtensionInterface.h"
#include "RendererPrivateUtils.h"
#include "SceneRendering.h"
#include "CurveSkinning/CurveSkinningDataProvider.h"

#define UE_API RENDERER_API

class FCurveSkinningSceneParameters;
class FPrimitiveSceneProxy;

class FCurveSkinningSceneExtension : public ISceneExtension, public ICurveSkinningSceneExtension
{
	DECLARE_SCENE_EXTENSION(RENDERER_API, FCurveSkinningSceneExtension);
	struct FHeaderData;

public:
	class FUpdater : public ISceneExtensionUpdater
	{
		DECLARE_SCENE_EXTENSION_UPDATER(FUpdater, FCurveSkinningSceneExtension);

	public:
		FUpdater(FCurveSkinningSceneExtension& InSceneData);

		virtual void End() override;
		virtual void PreSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePreUpdateChangeSet& ChangeSet) override;
		virtual void PostSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePostUpdateChangeSet& ChangeSet) override;

	private:
		FCurveSkinningSceneExtension* SceneData = nullptr;
		TArray<int32, FSceneRenderingArrayAllocator> AllocationUpdateList;
		TArray<int32, FSceneRenderingArrayAllocator> HeaderDataUpdateList;
		bool bForceFullUpload = false;
	};

	class FRenderer : public ISceneExtensionRenderer
	{
		DECLARE_SCENE_EXTENSION_RENDERER(FRenderer, FCurveSkinningSceneExtension);

	public:
		FRenderer(FSceneRendererBase& InSceneRenderer, FCurveSkinningSceneExtension& InSceneData)
			: ISceneExtensionRenderer(InSceneRenderer)
			, SceneData(&InSceneData)
		{}

		virtual void UpdateSceneUniformBuffer(FRDGBuilder& GraphBuilder, FSceneUniformBuffer& Buffer) override;

	private:
		static void AddDebugPass(FRDGBuilder& GraphBuilder, const TArray<FViewInfo>& Views, FCurveSkinningSceneExtension* SceneData, FSceneUniformBuffer& SceneUniformBuffer);
		FCurveSkinningSceneExtension* SceneData = nullptr;
	};

	friend class FUpdater;
	friend class FRenderer;

	explicit FCurveSkinningSceneExtension(FScene& InScene);
	virtual ~FCurveSkinningSceneExtension();

	// ~ Begin ISceneExtension
	static bool ShouldCreateExtension(FScene& InScene);
	virtual void InitExtension(FScene& InScene) override;
	virtual ISceneExtensionUpdater* CreateUpdater() override;
	virtual ISceneExtensionRenderer* CreateRenderer(FSceneRendererBase& InSceneRenderer, const FEngineShowFlags& EngineShowFlags) override;
	// ~End ISceneExtension

	//~ Begin ICurveSkinningSceneExtension interface (push-registration entry points).
	UE_API virtual FCurveSkinningHandle Register(FPrimitiveSceneProxy* OwningProxy, TUniquePtr<FCurveSkinningSceneExtensionProxy>&& InData) override;
	UE_API virtual void Unregister(FCurveSkinningHandle Handle) override;
	//~ End ICurveSkinningSceneExtension interface

private:
	enum ETask : uint32
	{
		FreeBufferSpaceTask,
		InitHeaderDataTask,
		AllocBufferSpaceTask,

		NumTasks
	};

	// Opaque key for sharing rest-position buffer slots across primitives using the same source asset.
	struct FRestDataKey
	{
		const void* AssetPtr = nullptr; // Never dereferenced it. Only used as key
		bool operator==(const FRestDataKey& Other) const { return AssetPtr == Other.AssetPtr; }
		friend SIZE_T GetTypeHash(const FRestDataKey& Key) { return ::GetTypeHash(Key.AssetPtr); }
	};

	struct FSharedRestData
	{
		uint32 RestBufferOffset = INDEX_NONE;
		uint32 RestBufferCount  = 0;
		int32  RefCount         = 0;
	};

	struct FHeaderData
	{
		FPrimitiveSceneInfo*               PrimitiveSceneInfo        = nullptr;
		// Owned by the extension. The proxy that originally pushed the data via
		// ICurveSkinningSceneExtension::RegisterPending() no longer holds a pointer.
		TUniquePtr<FCurveSkinningSceneExtensionProxy> Proxy;
		FGuid                              ProviderId;
		uint32                             InstanceSceneDataOffset   = 0;
		uint32                             NumInstanceSceneDataEntries = 0;
		uint32                             RestBufferOffset          = INDEX_NONE;		// Offset in "float"
		uint32                             RestBufferCount           = 0;				// Count of "float"
		uint32                             DeformedBufferOffset      = INDEX_NONE;  	// Offset in "float"
		uint32                             DeformedBufferCount       = 0;				// Count of "float"
		uint16                             NumCurves                 = 0;
		uint8                              NumPointsPerCurve         = 0;
		EDirtyCurveData                    DirtyCurveData            = EDirtyCurveData::All;
		uint8                              CurrentSlot          : 1  = 0;
		uint8                              bIsAllocated         : 1  = false;

		FCurveSkinningHeader Pack() const
		{
			const uint32 FloatsPerSlot = (uint32)NumCurves * (uint32)NumPointsPerCurve * CURVESKINNING_FLOATS_PER_POINT;

			FCurveSkinningHeader Output;
			Output.RestBufferOffset         = RestBufferOffset;
			Output.DeformedBufferOffset     = DeformedBufferOffset + (CurrentSlot  * FloatsPerSlot);
			Output.PrevDeformedBufferOffset = DeformedBufferOffset + (!CurrentSlot * FloatsPerSlot);
			Output.NumCurves                = NumCurves;
			Output.NumPointsPerCurve        = NumPointsPerCurve;
			Output.CurrentSlot              = CurrentSlot;
			Output.Unused                   = 0;
			Output.Unused2                  = 0;
			return Output;
		}

		void Validate() const
		{
			const uint32 FloatsPerSlot = (uint32)NumCurves * (uint32)NumPointsPerCurve * CURVESKINNING_FLOATS_PER_POINT;
			check(RestBufferOffset     <= CURVESKINNING_BUFFER_REST_OFFSET_MAX);
			check(DeformedBufferOffset +  FloatsPerSlot * 2 <= CURVESKINNING_BUFFER_DEFORMED_OFFSET_MAX);
			check(NumCurves            <= CURVESKINNING_MAX_CURVES);
			check(NumPointsPerCurve    <= CURVESKINNING_MAX_POINTS_PER_CURVE);
		}
	};

	class FBuffers
	{
	public:
		FBuffers();

		TPersistentByteAddressBuffer<FCurveSkinningHeader> HeaderDataBuffer;
		TPersistentByteAddressBuffer<float>                RestPositionBuffer;
		TPersistentByteAddressBuffer<float>                DeformedPositionBuffer;
	};

	class FUploader
	{
	public:
		TByteAddressBufferScatterUploader<FCurveSkinningHeader> HeaderDataUploader;
		TByteAddressBufferScatterUploader<float>                RestPositionUploader;
		TByteAddressBufferScatterUploader<float>                DeformedPositionUploader;
	};

	bool IsEnabled() const { return Buffers.IsValid(); }
	void SetEnabled(bool bEnabled);

	void FinishCurveSkinningBufferUpload(
		FRDGBuilder& GraphBuilder,
		FCurveSkinningSceneParameters* OutParams = nullptr,
		bool bUpdateStats = false);

	void PerformCurveSkinning(
		FCurveSkinningSceneParameters& Parameters,
		FRDGBuilder& GraphBuilder,
		const FGameTime& CurrentTime);

	// Advance the curve-simulation state by one frame: flip every allocated primitive's
	// double-buffer slot and re-arm its dirty mask so the next frame's solver runs.
	// Called once per render-thread frame from FRenderer::UpdateSceneUniformBuffer,
	// immediately after PerformCurveSkinning, inside the same frame-gate.
	void AdvanceSimulationFrame();

	bool ProcessBufferDefragmentation();

	void AllocSpaceForPrimitive(FHeaderData& HeaderData, FUploader& InUploader);
	void FreeSpaceForPrimitive(FHeaderData& HeaderData);

	// --------------------------------------------------------------------------------
	// Push-registration state (written from RegisterPending/Unregister on the RT,
	// drained from FUpdater::Pre/PostSceneUpdate on the RT). All four sites run on
	// the render thread and are serialized via the RT command queue, so no locking
	// is required. See ICurveSkinningSceneExtension comments for the threading model.
	// --------------------------------------------------------------------------------

	// One pending entry per proxy that has called RegisterPending() but whose matching
	// FPrimitiveSceneInfo has not yet shown up in PostSceneUpdate's AddedPrimitiveSceneInfos.
	// Keyed by proxy pointer for the FPSI -> pending join in PostSceneUpdate. The data
	// is owned here until ownership is transferred into FHeaderData::Proxy at finalize time.
	struct FPendingRegistration
	{
		TUniquePtr<FCurveSkinningSceneExtensionProxy> Data;
		FCurveSkinningHandle                          Handle;
	};
	TMap<FPrimitiveSceneProxy*, FPendingRegistration> PendingRegistrations;

	// PersistentIndex entries queued by Unregister() for entries that were already finalized.
	// Drained by PreSceneUpdate's FreeBufferSpaceTask.
	TArray<int32>                                     PendingFreeIndices;

	// Handle table — translates an FCurveSkinningHandle back to either a pending entry
	// (PersistentIndex == INDEX_NONE) or a finalized HeaderData slot.
	struct FHandleSlot
	{
		FPrimitiveSceneProxy* Proxy           = nullptr;
		int32                 PersistentIndex = INDEX_NONE;
		uint32                Generation      = 0;
		bool                  bInUse          = false;
	};
	// Handle table for ICurveSkinningSceneExtension::Register / Unregister.
	//
	// Slots are NEVER removed from this array once allocated — Unregister only flips
	// bInUse and pushes the index onto FreeHandleSlotIndices. Register reuses an index
	// from the free-list (or grows the array if empty) and bumps Slot.Generation. This
	// way Generation monotonically increases across the slot's lifetime, so a stale
	// handle {Index, Generation} can be distinguished from a newer {Index, Generation+N}
	// handle at the same recycled index.
	TArray<FHandleSlot>                    HandleSlots;
	TArray<int32>                          FreeHandleSlotIndices;

	FSpanAllocator                         RestAllocator;
	FSpanAllocator                         DeformedAllocator;
	TSparseArray<FHeaderData>              HeaderDatas;
	TSet<int32>                            AllocatedHeaderDataIndices;
	TMap<FRestDataKey, FSharedRestData>    SharedRestDatas;
	TUniquePtr<FBuffers>                   Buffers;
	TUniquePtr<FUploader>                  Uploader;
	TStaticArray<UE::Tasks::FTask,NumTasks>TaskHandles;
	uint32                                 UpdateCounter          = 0;
	uint32                                 LastUpdateFrameNumber  = ~0;
};

#undef UE_API