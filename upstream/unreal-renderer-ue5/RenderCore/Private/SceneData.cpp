// Copyright Epic Games, Inc. All Rights Reserved.

#include "SceneData.h"
#include "Containers/Map.h"
#include "HAL/CriticalSection.h"
#include <atomic>

namespace SceneData
{
#if SCENE_DATA_DEBUG_ACCESS_RACES_DYNAMIC

template <typename TransformFunc>
void FBaseDataArray::TransitionState(TransformFunc&& Transform) const
{
	uint64 Current = AccessState.load(std::memory_order_acquire);
	for (;;)
	{
		const uint32 WriterId    = UnpackWriter(Current);
		const uint32 ReaderId    = UnpackReaderContext(Current);
		const uint32 WriterCount = UnpackWriterCount(Current);
		const uint32 ReaderCount = UnpackReaderCount(Current);
		const uint64 Desired = Transform(WriterId, ReaderId, WriterCount, ReaderCount);
		if (AccessState.compare_exchange_weak(Current, Desired,
				std::memory_order_acq_rel /* publish on success */,
				std::memory_order_acquire /* reload on failure */))
		{
			return;
		}
	}
}

void FBaseDataArray::BeginRead(uint32 DebugId) const
{
	check(DebugId != 0);
	TransitionState([DebugId](uint32 WriterId, uint32 ReaderId, uint32 WriterCount, uint32 ReaderCount) -> uint64
	{
		check(ReaderCount < CountMask);                           // overflow guard
		if (WriterCount == 0)
		{
			// No writer: track ReaderId iff all current readers share one id.
			const uint32 NewReaderId =
				(ReaderCount == 0)       ? DebugId :              // first reader
				(ReaderId    == DebugId) ? DebugId :              // still all same
					                        0;                     // now mixed
			return Pack(0, NewReaderId, 0, ReaderCount + 1);
		}
		// Writer active: any same-id reader allowed (no other id can BeginRead here).
		checkf(WriterId == DebugId, TEXT("WriterId=%d, DebugId=%d"), WriterId, DebugId);
		checkf(ReaderCount == 0 || ReaderId == DebugId, TEXT("ReaderCount=%d,ReaderId=%dDebugId=%d"),ReaderCount,ReaderId,DebugId);
		return Pack(WriterId, DebugId, WriterCount, ReaderCount + 1);
	});
}

void FBaseDataArray::EndRead(uint32 DebugId) const
{
	check(DebugId != 0);
	TransitionState([DebugId](uint32 WriterId, uint32 ReaderId, uint32 WriterCount, uint32 ReaderCount) -> uint64
	{
		check(ReaderCount > 0);
		if (WriterCount > 0)
		{
			checkf(WriterId == DebugId, TEXT("WriterId=%d, DebugId=%d"), WriterId, DebugId);
			checkf(ReaderId == DebugId, TEXT("ReaderId=%d, DebugId=%d"), ReaderId, DebugId); // writer-active ⇒ readers are all `DebugId`
		}
		// Clear ReaderId only when the last reader leaves; otherwise keep as-is
		// (all-same stays all-same; mixed stays mixed — we don't try to recover from mixed).
		const uint32 NewReaderId = (ReaderCount == 1) ? 0 : ReaderId;
		return Pack(WriterId, NewReaderId, WriterCount, ReaderCount - 1);
	});
}

void FBaseDataArray::BeginWrite(uint32 DebugId)
{
	check(DebugId != 0);
	TransitionState([DebugId](uint32 WriterId, uint32 ReaderId, uint32 WriterCount, uint32 ReaderCount) -> uint64
	{
		check(WriterCount < CountMask);                           // overflow guard
		if (WriterCount > 0)
		{
			// Nested write — same id.
			check(WriterId == DebugId);
			return Pack(WriterId, ReaderId, WriterCount + 1, ReaderCount);
		}
		// First write: writer slot must be free; promotion allowed across any number of same-id readers.
		checkf(WriterId    == 0, TEXT("WriterId=%d"), WriterId);
		checkf(ReaderCount == 0 || ReaderId == DebugId, TEXT("ReaderCount=%d,ReaderId=%dDebugId=%d"),ReaderCount,ReaderId,DebugId);
		return Pack(DebugId, ReaderId, 1, ReaderCount);
	});
}

void FBaseDataArray::EndWrite(uint32 DebugId)
{
	check(DebugId != 0);
	TransitionState([DebugId](uint32 WriterId, uint32 ReaderId, uint32 WriterCount, uint32 ReaderCount) -> uint64
	{
		check(WriterCount > 0);
		check(WriterId == DebugId);                               // matching writer
		if (WriterCount > 1)
		{
			// Inner end — keep writer slot held.
			return Pack(WriterId, ReaderId, WriterCount - 1, ReaderCount);
		}
		// Outermost end — release writer slot; any remaining readers must be same id.
		checkf(ReaderCount == 0 || ReaderId == DebugId, TEXT("ReaderCount=%d,ReaderId=%dDebugId=%d"),ReaderCount,ReaderId,DebugId);
		return Pack(0, ReaderId, 0, ReaderCount);
	});
}
#endif

FBaseAccessContext::FBaseAccessContext(FDependencyManager& InBuilder)
	: Builder(InBuilder)
	, ContextId(1u)
{
}

FBaseAccessContext::FBaseAccessContext(FDependencyManager& InBuilder, FChildTag)
	: Builder(InBuilder)
	, ContextId(++InBuilder.CurrentContextId)
{
	EnabledReaders.SetNum(Builder.ResourceHeadState.Num(), false);
	EnabledWriters.SetNum(Builder.ResourceHeadState.Num(), false);
}


void FBaseAccessContext::BeginReadAccess(int32 ResourceId, ESyncMode SyncMode)
{
	if (IsRoot())
	{
		if (ProcessingStage == EProcessingStage::ReadOnlyUntracked )
		{
			return;
		}
		bool bWaited = Builder.WaitForResoureRead(ResourceId);
		
		// check that we did not need to wait in a validate-only test
		checkf(SyncMode == ESyncMode::SyncAndValidate || !bWaited, TEXT("Wait on access incurrect for unsynced access, missing explicit Wait."));
	}
	else
	{
		check(EnabledReaders[ResourceId] || EnabledWriters[ResourceId]);
	}
}

void FBaseAccessContext::BeginWriteAccess(int32 ResourceId, ESyncMode SyncMode)
{
	if (IsRoot())
	{
		bool bWaited = Builder.WaitForResoureWrite(ResourceId);
		// check that we did not need to wait in a validate-only test
		checkf(SyncMode == ESyncMode::SyncAndValidate || !bWaited, TEXT("Wait on access incurrect for unsynced access, missing explicit Wait."));
	}
	else
	{
		check(EnabledWriters[ResourceId]);
	}
}

namespace
{
	thread_local FExternalAccessContextContainer* GExternalAccessContexts = nullptr;
}

FExternalAccessContextContainer*& GetExternalAccessContextsRef()
{
	return GExternalAccessContexts;
}

UE::FInheritedContextExtension& GetExternalAccessExtension()
{
	static UE::FInheritedContextExtension Ext = UE::MakeInheritedContextExtension(&GetExternalAccessContextsRef);
	return Ext;
}

#if SCENE_DATA_DEBUG_ACCESS_RACES_DYNAMIC

namespace
{
	thread_local uint16 GDebugId          = 0;   // scope-pushed value (0 = unset)
	thread_local uint16 GThreadDebugIndex = 0;   // lazy per-thread index (0 = unassigned)
	std::atomic<uint16> GThreadDebugIndexCounter{ 1 };   // first thread gets 1; 0 reserved
}

uint16& GetDebugIdRef()
{
	return GDebugId;
}

uint16 GetEffectiveDebugId()
{
	if (GDebugId != 0)
	{
		return GDebugId;
	}
	if (GThreadDebugIndex == 0)
	{
		const uint16 NewIndex = GThreadDebugIndexCounter.fetch_add(1, std::memory_order_relaxed);
		checkf(NewIndex != 0,
			TEXT("SceneData thread-local debug-id counter exhausted (>65534 threads); widen the field"));
		GThreadDebugIndex = NewIndex;
	}
	return GThreadDebugIndex;
}

UE::FInheritedContextExtension& GetDebugIdExtension()
{
	static UE::FInheritedContextExtension Ext = UE::MakeInheritedContextExtension(&GetDebugIdRef);
	return Ext;
}

// Always-active so every UE::Tasks::Launch / ParallelFor captures GDebugId via inherited-context.
static UE::FInheritedContextExtensionScope GDebugIdGlobalScope(GetDebugIdExtension());

#endif

uint32 FExternalAccessContextContainer::AllocateManagerTypeIndex(const TCHAR* ManagerTypeName)
{
	static FCriticalSection IndexLock;
	static TMap<FString, uint32> NameToIndex;
	static uint32 NextIndex = 0;

	FScopeLock Lock(&IndexLock);
	const FString Key(ManagerTypeName);
	if (const uint32* Found = NameToIndex.Find(Key))
	{
		return *Found;
	}
	const uint32 Index = NextIndex++;
	checkf(Index < (uint32)MaxManagerTypes, TEXT("Raise SceneData::MaxManagerTypes (current=%d, type=%s)"), MaxManagerTypes, ManagerTypeName);
	NameToIndex.Emplace(Key, Index);
	return Index;
}

void FDependencyManager::Flush()
{ 
	check(IsInRenderingThread());
	UE::Tasks::Wait(Tasks);
	ResetTrackingState();
}

void FDependencyManager::FrameEnd()
{
	Flush();
}

void FDependencyManager::WaitAndClear(const FResourceEvent& Event)
{
	check(IsInRenderingThread());
	UE::Tasks::FTask& Task = Tasks[Event.TaskId];
	if (Task.IsValid())
	{
		Task.Wait();
		Task = UE::Tasks::FTask();
		ActiveTaskCount -= 1;
	}
}

void FDependencyManager::ResetTrackingState()
{
	check(IsInRenderingThread());
	// check(ActiveContextCount == 0);

	// Pre-condition: Flush() has already Wait()'d the Tasks array, so no live tasks hold pointers
	// into ChildContextStorage. Safe to destruct everything allocated this frame.
	ChildContextStorage.BulkDelete();

	ResourceEvents.Reset();
	ResourceHeadState.Reset();
	Tasks.Reset();
	CurrentContextId = 1u;
	ActiveTaskCount = 0;
}

bool FDependencyManager::bSuppressRootContextValidation = true;
bool FBaseDataManager::bLogRootContextWaits = false;
bool FBaseDataManager::bLogOutsideUpdateWrites = false;

bool FDependencyManager::WaitForResoureRead(int32 ResourceId)
{
	// Off-RT call to root context (only path that leads to this function), suppress checks to allow legacy tasks to access data unsafely.
	if (bSuppressRootContextValidation && !IsInRenderingThread())
	{
		return false;
	}

	check(IsInRenderingThread());

	if (!ResourceHeadState.IsValidIndex(ResourceId))
	{
		return false;
	}

	if (UNLIKELY(FBaseDataManager::bLogRootContextWaits))
	{
		FDebug::DumpStackTraceToLog(TEXT("Root context wait-on-demand triggered (read)"), ELogVerbosity::Warning);
	}

	FResourceState& State = ResourceHeadState[ResourceId];
	if (State.WaitState == EWaitState::None)
	{
		return false;
	}

	// Only need to wait if there's a write event, reads are allowed to run in parallel.
	const bool bNeedToWait = State.WaitState == EWaitState::Write;
	if (bNeedToWait)
	{
		check(State.LastWriteEventId != INDEX_NONE);
		FResourceEvent& LastWriteEvent = ResourceEvents[State.LastWriteEventId];
		check(!LastWriteEvent.bReadEvent);

		WaitAndClear(LastWriteEvent);
		State.WaitState = EWaitState::None;
	}

	check(ActiveTaskCount >= 0);
	// No more outstanding tasks, reset the tracking state to early out all testing logic
	if (ActiveTaskCount == 0)
	{
		ResetTrackingState();
	}
	return bNeedToWait;
}


bool FDependencyManager::WaitForResoureWrite(int32 ResourceId)
{
	if (bSuppressRootContextValidation && !IsInRenderingThread())
	{
		return false;
	}
	check(IsInRenderingThread());

	if (!ResourceHeadState.IsValidIndex(ResourceId))
	{
		return false;
	}

	if (UNLIKELY(FBaseDataManager::bLogRootContextWaits))
	{
		FDebug::DumpStackTraceToLog(TEXT("Root context wait-on-demand triggered (write)"), ELogVerbosity::Warning);
	}

	FResourceState& State = ResourceHeadState[ResourceId];
	if (State.WaitState == EWaitState::None)
	{
		return false;
	}

	if (State.WaitState == EWaitState::Read)
	{
		check(State.LastReadEventId > State.LastWriteEventId);

		for (int32 EventId = State.LastReadEventId; EventId > State.LastWriteEventId; )
		{
			const FResourceEvent& Event = ResourceEvents[EventId];
			check(Event.bReadEvent);
			WaitAndClear(Event);
			EventId = Event.PreviousEventId;
		}
	}
	else if (State.WaitState == EWaitState::Write)
	{
		check(State.LastWriteEventId != INDEX_NONE);
		WaitAndClear(ResourceEvents[State.LastWriteEventId]);
	}
	// Waited for all the events for this resource.
	State.WaitState = EWaitState::None;

	check(ActiveTaskCount >= 0);
	// No more outstanding tasks, reset the tracking state to early out all testing logic
	if (ActiveTaskCount == 0)
	{
		ResetTrackingState();
	}
	return true;
}

void FDependencyManager::ProcessDependencies(
	FBaseAccessContext& ChildDepCtx,
	TArray<UE::Tasks::FTask>& Prerequisites,
	int32 TaskId,
	const TArray<FParameter>& ReadDeps,
	const TArray<FParameter>& WriteDeps,
	bool bIsRunningAsync)
{
	check(IsInRenderingThread());

	ResourceEvents.Reserve(ResourceEvents.Num() + ReadDeps.Num() + WriteDeps.Num());

	auto AddPrereq = [&](UE::Tasks::FTask Task, int32 ResourceId, EAccessFlags Flags)
	{
		if (EnumHasAnyFlags(Flags, EAccessFlags::Dependency))
		{
			Prerequisites.AddUnique(Task);
		}
	};

	// Reads — depend on the latest writer (parallel readers don't see each other).
	for (FParameter Parameter : ReadDeps)
	{
		const FResourceState& State = ResourceHeadState[Parameter.ResourceId];

		if (State.WaitState == EWaitState::Write)
		{
			check(State.LastWriteEventId != INDEX_NONE);

			const FResourceEvent& LastWriteEvent = ResourceEvents[State.LastWriteEventId];
			check(!LastWriteEvent.bReadEvent);
			AddPrereq(Tasks[LastWriteEvent.TaskId], Parameter.ResourceId, Parameter.AccessFlags);
		}
	}

	// Writes — depend on every reader since the last writer, OR on the last writer if there
	// have been no readers since.
	for (FParameter Parameter : WriteDeps)
	{
		const FResourceState& State = ResourceHeadState[Parameter.ResourceId];
		if (State.WaitState == EWaitState::Read)
		{
			check(State.LastReadEventId > State.LastWriteEventId);

			for (int32 EventId = State.LastReadEventId; EventId > State.LastWriteEventId; )
			{
				const FResourceEvent& Event = ResourceEvents[EventId];
				check(Event.bReadEvent);
				AddPrereq(Tasks[Event.TaskId], Parameter.ResourceId, Parameter.AccessFlags);
				EventId = Event.PreviousEventId;
			}
		}
		else if (State.WaitState == EWaitState::Write)
		{
			check(State.LastWriteEventId != INDEX_NONE);

			const FResourceEvent& LastWriteEvent = ResourceEvents[State.LastWriteEventId];
			check(!LastWriteEvent.bReadEvent);
			AddPrereq(Tasks[LastWriteEvent.TaskId], Parameter.ResourceId, Parameter.AccessFlags);
		}
	}

	auto AppendReadEvent = [&](int32 ResourceId)
	{
		FResourceState& State = ResourceHeadState[ResourceId];
		int32 NewEventId = ResourceEvents.Add(FResourceEvent{TaskId, State.LastReadEventId, true});
		State.LastReadEventId = NewEventId;
		State.WaitState = bIsRunningAsync ? EWaitState::Read : EWaitState::None;

	};

	// Append new events (after gathering, so a task doesn't depend on itself).
	for (FParameter Parameter : ReadDeps)
	{
		ChildDepCtx.EnabledReaders[Parameter.ResourceId] = true;
		AppendReadEvent(Parameter.ResourceId);
	}

	for (FParameter Parameter : WriteDeps)
	{
		ChildDepCtx.EnabledWriters[Parameter.ResourceId] = true;

		// Add as a read event (to make it _not_ a barrier to subsequent reads)
		if (EnumHasAnyFlags(Parameter.AccessFlags, EAccessFlags::RecordAsReadEvent))
		{
			AppendReadEvent(Parameter.ResourceId);
		}
		else
		{
			FResourceState& State = ResourceHeadState[Parameter.ResourceId];
			int32 NewEventId = ResourceEvents.Add(FResourceEvent{TaskId, State.LastWriteEventId, false});
			State.LastWriteEventId = NewEventId;
			State.WaitState = bIsRunningAsync ? EWaitState::Write : EWaitState::None;
		}
	}
}

void FDependencyManager::OnTaskLaunched(const UE::Tasks::FTask& Task)
{
	check(IsInRenderingThread());
	Tasks.Emplace(Task);
	ActiveTaskCount += 1;
}

} // namespace SceneData
