// Copyright Epic Games, Inc. All Rights Reserved.

#include "RenderDeferredCleanup.h"
#include "HAL/IConsoleManager.h"
#include "RenderingThread.h"

static bool bEnablePendingCleanupObjectsCommandBatching = true;
static FAutoConsoleVariableRef CVarEnablePendingCleanupObjectsCommandBatching(
	TEXT("g.bEnablePendingCleanupObjectsCommandBatching"),
	bEnablePendingCleanupObjectsCommandBatching,
	TEXT("Enable batching PendingCleanupObjects destruction.")
);

#if WITH_EDITOR || IS_PROGRAM

// mainly concerned about the cooker here, but anyway, the editor can run without a frame for a very long time (hours) and we do not have enough lock free links. 

/** The set of deferred cleanup objects which are pending cleanup. */
TArray<FDeferredCleanupInterface*> PendingCleanupObjectsList;
FCriticalSection PendingCleanupObjectsListLock;

FPendingCleanupObjects::FPendingCleanupObjects()
{
	check(IsInGameThread());
	{
		FScopeLock Lock(&PendingCleanupObjectsListLock);
		Exchange(CleanupArray, PendingCleanupObjectsList);
	}
}

FPendingCleanupObjects::~FPendingCleanupObjects()
{
	if (CleanupArray.Num())
	{
		QUICK_SCOPE_CYCLE_COUNTER(STAT_FPendingCleanupObjects_Destruct);

		const bool bBatchingEnabled = bEnablePendingCleanupObjectsCommandBatching;
		if (bBatchingEnabled)
		{
			StartRenderCommandFenceBundler();
		}
		for (int32 ObjectIndex = 0; ObjectIndex < CleanupArray.Num(); ObjectIndex++)
		{
			delete CleanupArray[ObjectIndex];
		}
		if (bBatchingEnabled)
		{
			StopRenderCommandFenceBundler();
		}
	}
}

void BeginCleanup(FDeferredCleanupInterface* CleanupObject)
{
	FScopeLock Lock(&PendingCleanupObjectsListLock);
	PendingCleanupObjectsList.Add(CleanupObject);
}

#else

/** The set of deferred cleanup objects which are pending cleanup. */
static TLockFreePointerListUnordered<FDeferredCleanupInterface, PLATFORM_CACHE_LINE_SIZE>	PendingCleanupObjectsList;

FPendingCleanupObjects::FPendingCleanupObjects()
{
	check(IsInGameThread());
	PendingCleanupObjectsList.PopAll(CleanupArray);
}

FPendingCleanupObjects::~FPendingCleanupObjects()
{
	if (CleanupArray.Num())
	{
		QUICK_SCOPE_CYCLE_COUNTER(STAT_FPendingCleanupObjects_Destruct);

		const bool bBatchingEnabled = bEnablePendingCleanupObjectsCommandBatching;
		if (bBatchingEnabled)
		{
			StartRenderCommandFenceBundler();
		}
		for (int32 ObjectIndex = 0; ObjectIndex < CleanupArray.Num(); ObjectIndex++)
		{
			delete CleanupArray[ObjectIndex];
		}
		if (bBatchingEnabled)
		{
			StopRenderCommandFenceBundler();
		}
	}
}

void BeginCleanup(FDeferredCleanupInterface* CleanupObject)
{
	PendingCleanupObjectsList.Push(CleanupObject);
}

#endif

FPendingCleanupObjects* GetPendingCleanupObjects()
{
	return new FPendingCleanupObjects;
}


