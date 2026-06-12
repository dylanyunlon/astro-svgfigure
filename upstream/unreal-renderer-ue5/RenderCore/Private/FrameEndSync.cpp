// Copyright Epic Games, Inc. All Rights Reserved.

#include "RenderingThread.h"
#include "RenderCommandFence.h"

TAutoConsoleVariable<int32> CVarAllowOneFrameThreadLag(
	TEXT("r.OneFrameThreadLag"),
	1,
	TEXT("Whether to allow the rendering thread to lag one frame behind the game thread (0: disabled, otherwise enabled)")
);

TAutoConsoleVariable<int32> CVarGTSyncType(
	TEXT("r.GTSyncType"),
	0,
	TEXT("Determines how the game thread syncs with the render thread, RHI thread and GPU.\n")
	TEXT("Syncing to the GPU swap chain flip allows for lower frame latency.\n")
	TEXT(" <= 0 - Sync the game thread with the N-1 render thread frame. Then sync with the N-m RHI thread frame where m is (2 + (-r.GTSyncType)) (i.e. negative values increase the amount of RHI thread overlap) (default = 0).\n")
	TEXT("    1 - Sync the game thread with the N-1 RHI thread frame.\n")
	TEXT("    2 - Sync the game thread with the GPU swap chain flip (only on supported platforms).\n"),
	ECVF_Default
);

DECLARE_CYCLE_STAT(TEXT("Frame Sync Time"), STAT_FrameSyncTime, STATGROUP_RenderThreadProcessing);

namespace FFrameEndSync
{
	using ESyncDepth = FRenderCommandFence::ESyncDepth;

	struct FRenderThreadFence
	{
		// Legacy game code assumes the game thread will never get further than 1 frame ahead of the render thread.
		// This fence is used to sync the game thread with the N-1 render thread frame.
		FRenderCommandFence Fence;

		FRenderThreadFence()
		{
			Fence.BeginFence(ESyncDepth::RenderThread);
		}

		~FRenderThreadFence()
		{
			Fence.Wait(true);
		}
	};
	TArray<FRenderThreadFence, TInlineAllocator<2>> RenderThreadFences;

	// Additional fences to await. These sync with either the RHI thread or swapchain,
	// and are used to prevent the game thread running too far ahead of presented frames.
	TArray<FRenderCommandFence, TInlineAllocator<3>> PipelineFences;

	void Sync(EFlushMode FlushMode)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE(Sync_RenderingThread);
		static bool bRecursive = false;
		TGuardValue<bool> RecursionGuard(bRecursive, true);

		if (RecursionGuard.GetOriginalValue())
		{
			// This is a recursive call to FFrameEndSync::Sync(). Use a standard render fence and do a full sync.
			FRenderCommandFence Fence;
			Fence.BeginFence();
			Fence.Wait();
			return;
		}

		bool bFullSync = FlushMode == EFlushMode::Threads;

		// The "r.OneFrameThreadLag" cvar forces a full sync, meaning the game thread will
		// not start work until all the rendering work for the previous frame has completed.
		bFullSync |= CVarAllowOneFrameThreadLag.GetValueOnAnyThread() <= 0;

		SCOPE_CYCLE_COUNTER(STAT_FrameSyncTime);

		check(IsInGameThread());

		// Always sync with the render thread (either current frame, or N-1 frame)
		RenderThreadFences.Emplace();
		while (RenderThreadFences.Num() > (bFullSync ? 0 : 1))
		{
			RenderThreadFences.RemoveAt(0);
		}

		// Insert an additional fence based on how we want to sync with the RHI thread / swapchain
		ESyncDepth SyncDepth;
		int32 NumFramesOverlap;

		int32 const GTSyncType = CVarGTSyncType.GetValueOnAnyThread();

		if (bFullSync)
		{
			SyncDepth = (GTSyncType >= 2 && FlushMode != EFlushMode::Threads)
				? ESyncDepth::Swapchain
				: ESyncDepth::RHIThread;

			NumFramesOverlap = 0;
		}
		else if (GTSyncType >= 2)
		{
			SyncDepth = ESyncDepth::Swapchain;
			NumFramesOverlap = 1;
		}
		else if (GTSyncType == 1)
		{
			SyncDepth = ESyncDepth::RHIThread;
			NumFramesOverlap = 1;
		}
		else
		{
			check(GTSyncType <= 0);

			// Modes <= 0 allows N frames of overlap with the RHI thread.
			SyncDepth = ESyncDepth::RHIThread;
			NumFramesOverlap = 2 + (-GTSyncType);
		}

		if (SyncDepth == ESyncDepth::Swapchain)
		{
			// Swapchain sync mode does not work when vsync is disabled. Fallback to RHI thread sync in that case.
			static auto CVarVsync = IConsoleManager::Get().FindConsoleVariable(TEXT("r.VSync"));
			check(CVarVsync != nullptr);

			if (CVarVsync->GetInt() == 0)
			{
				SyncDepth = ESyncDepth::RHIThread;
			}
		}

		PipelineFences.Emplace_GetRef().BeginFence(SyncDepth);

		// Don't process game thread tasks when flushing all threads. This can result in strange behavior where the game thread
		// is flushing the render thread and then gets pre-empted by another task that has an implicit dependency on the one
		// being processed.
		if (FlushMode == EFlushMode::EndFrame && !FTaskGraphInterface::Get().IsThreadProcessingTasks(ENamedThreads::GameThread))
		{
			// need to process gamethread tasks at least once a frame no matter what
			FTaskGraphInterface::Get().ProcessThreadUntilIdle(ENamedThreads::GameThread);
		}

		while (PipelineFences.Num() > NumFramesOverlap)
		{
			PipelineFences[0].Wait(true);
			PipelineFences.RemoveAt(0);
		}
	}
}
