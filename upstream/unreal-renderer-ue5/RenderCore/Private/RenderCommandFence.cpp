// Copyright Epic Games, Inc. All Rights Reserved.

#include "RenderCommandFence.h"
#include "HAL/ExceptionHandling.h"
#include "HAL/ThreadHeartBeat.h"
#include "HAL/ThreadManager.h"
#include "Misc/CoreStats.h"
#include "Misc/OutputDeviceRedirector.h"
#include "Misc/TimeGuard.h"
#include "RenderCore.h"
#include "RenderingThread.h"
#include "RenderResource.h"
#include "RenderThreadTimeoutControl.h"
#include "RHIUtilities.h"

bool GRenderCommandFenceBundling = true;
FAutoConsoleVariableRef CVarRenderCommandFenceBundling(
	TEXT("r.RenderCommandFenceBundling"),
	GRenderCommandFenceBundling,
	TEXT("Controls whether render command fences are allowed to be batched.\n")
	TEXT(" 0: disabled;\n")
	TEXT(" 1: enabled (default);\n"),
	ECVF_Default);

static int32 GTimeToBlockOnRenderFence = 1;
static FAutoConsoleVariableRef CVarTimeToBlockOnRenderFence(
	TEXT("g.TimeToBlockOnRenderFence"),
	GTimeToBlockOnRenderFence,
	TEXT("Number of milliseconds the game thread should block when waiting on a render thread fence.")
);


static int32 GTimeoutForBlockOnRenderFence = 120000;
static FAutoConsoleVariableRef CVarTimeoutForBlockOnRenderFence(
	TEXT("g.TimeoutForBlockOnRenderFence"),
	GTimeoutForBlockOnRenderFence,
	TEXT("Number of milliseconds the game thread should wait before failing when waiting on a render thread fence.")
);

std::atomic<int> GTimeoutSuspendCount;

void SuspendRenderThreadTimeout()
{
	++GTimeoutSuspendCount;
}

void ResumeRenderThreadTimeout()
{
	--GTimeoutSuspendCount;

	check(GTimeoutSuspendCount >= 0);
}

bool IsRenderThreadTimeoutSuspended()
{
	return GTimeoutSuspendCount > 0;
}

static void HandleRenderTaskHang(uint32 ThreadThatHung, double HangDuration)
{
	// Get the name of the hung thread
	FString ThreadName = FThreadManager::GetThreadName(ThreadThatHung);
	if (ThreadName.IsEmpty())
	{
		ThreadName = FString::Printf(TEXT("unknown thread (%u)"), ThreadThatHung);
	}

#if !PLATFORM_WINDOWS || (PLATFORM_USE_MINIMAL_HANG_DETECTION && 1)
	UE_LOGF(LogRendererCore, Fatal, "GameThread timed out waiting for %ls after %.02f secs", *ThreadName, HangDuration);
#else
	// Capture the stack in the thread that hung
	static const int32 MaxStackFrames = 100;
	uint64 StackFrames[MaxStackFrames];
	int32 NumStackFrames = FPlatformStackWalk::CaptureThreadStackBackTrace(ThreadThatHung, StackFrames, MaxStackFrames);

	// Convert the stack trace to text
	TArray<FString> StackLines;
	for (int32 Idx = 0; Idx < NumStackFrames; Idx++)
	{
		ANSICHAR Buffer[1024];
		Buffer[0] = '\0';
		FPlatformStackWalk::ProgramCounterToHumanReadableString(Idx, StackFrames[Idx], Buffer, sizeof(Buffer));
		StackLines.Add(Buffer);
	}

	// Dump the callstack and the thread name to log
	FString StackTrimmed;
	UE_LOGF(LogRendererCore, Error, "GameThread timed out waiting for %ls after %.02f seconds:", *ThreadName, HangDuration);
	for (int32 Idx = 0; Idx < StackLines.Num(); Idx++)
	{
		UE_LOGF(LogRendererCore, Error, "  %ls", *StackLines[Idx]);
		if (StackTrimmed.Len() < 512)
		{
			StackTrimmed += TEXT("  ");
			StackTrimmed += StackLines[Idx];
			StackTrimmed += LINE_TERMINATOR;
		}
	}

	const FString ErrorMessage = FString::Printf(TEXT("GameThread timed out waiting for %s after %.02f seconds:%s%s%sCheck log for full callstack."),
		*ThreadName, HangDuration, LINE_TERMINATOR, *StackTrimmed, LINE_TERMINATOR);

	GLog->Panic();
	FCoreDelegates::OnHandleSystemHang.Broadcast();
	ReportHang(*ErrorMessage, StackFrames, NumStackFrames, ThreadThatHung);
	if (FApp::CanEverRender())
	{
		FPlatformMisc::MessageBoxExt(EAppMsgType::Ok,
			*NSLOCTEXT("MessageDialog", "ReportHangError_Body", "The application has hung and will now close. We apologize for the inconvenience.").ToString(),
			*NSLOCTEXT("MessageDialog", "ReportHangError_Title", "Application Hang Detected").ToString());
	}
	
	GIsCriticalError = true;
	FPlatformMisc::RequestExit(true, TEXT("GameThreadWaitForTask"));
#endif
}

/**
 * Block the game thread waiting for a task to finish on the rendering thread.
 */
static void GameThreadWaitForTask(const UE::Tasks::FTask& Task, bool bEmptyGameThreadTasks = false)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(GameThreadWaitForTask);
	SCOPE_TIME_GUARD(TEXT("GameThreadWaitForTask"));

	check(IsInGameThread());
	check(Task.IsValid());

	if (!Task.IsCompleted())
	{
		SCOPE_CYCLE_COUNTER(STAT_GameIdleTime);
		{
			static int32 NumRecursiveCalls = 0;

			// Check for recursion. It's not completely safe but because we pump messages while 
			// blocked it is expected.
			NumRecursiveCalls++;
			if (NumRecursiveCalls > 1)
			{
				UE_LOGF(LogRendererCore, Warning, "FlushRenderingCommands called recursively! %d calls on the stack.", NumRecursiveCalls);
			}
			if (NumRecursiveCalls > 1 || FTaskGraphInterface::Get().IsThreadProcessingTasks(ENamedThreads::GameThread))
			{
				bEmptyGameThreadTasks = false; // we don't do this on recursive calls or if we are at a blueprint breakpoint
			}

			// Check rendering thread health needs to be called from time to
			// time in order to pump messages, otherwise the RHI may block
			// on vsync causing a deadlock. Also we should make sure the
			// rendering thread hasn't crashed :)
			bool bDone;
			uint32 WaitTime = FMath::Clamp<uint32>(GTimeToBlockOnRenderFence, 0, 33);

			// Use a clamped clock to prevent taking into account time spent suspended.
			FThreadHeartBeatClock RenderThreadTimeoutClock((4 * WaitTime) / 1000.0);
			const double StartTime = RenderThreadTimeoutClock.Seconds();
			const double EndTime = StartTime + (GTimeoutForBlockOnRenderFence / 1000.0);

			bool bRenderThreadEnsured = FDebug::IsEnsuring();

			static bool bDisabled = FParse::Param(FCommandLine::Get(), TEXT("nothreadtimeout"));

			// Creating the wait task manually is a workaround for the problem of FTast::Wait creating
			// a separate wait task and event object on each call. It's a problem because we may call
			// Wait it in the loop below many times during long frame syncs (e.g. when using GPU profilers)
			// which would create thousands of such objects and run out of system resources.
			FSharedEventRef CompletionEvent;

			UE::Tasks::Launch(
				TEXT("Waiting Task (FrameSync)"),
				[CompletionEvent] { CompletionEvent->Trigger(); },
				Task,
				LowLevelTasks::ETaskPriority::Default,
				UE::Tasks::EExtendedTaskPriority::Inline,
				UE::Tasks::ETaskFlags::None
			);

			do
			{
				CheckRenderingThreadHealth();
				if (bEmptyGameThreadTasks)
				{
					// process gamethread tasks if there are any
					FTaskGraphInterface::Get().ProcessThreadUntilIdle(ENamedThreads::GameThread);
				}
				bDone = CompletionEvent->Wait(FTimespan::FromMilliseconds(WaitTime));

				RenderThreadTimeoutClock.Tick();

				const bool bOverdue = RenderThreadTimeoutClock.Seconds() >= EndTime && FThreadHeartBeat::Get().IsBeating();

				// track whether the thread ensured, if so don't do timeout checks
				bRenderThreadEnsured |= FDebug::IsEnsuring();

#if !WITH_EDITOR
#if !PLATFORM_IOS && !PLATFORM_MAC // @todo Metal: Timeout isn't long enough...
				// editor threads can block for quite a while... 
				if (!bDone && !bRenderThreadEnsured)
				{
					if (bOverdue && !bDisabled && !IsRenderThreadTimeoutSuspended() && !FPlatformMisc::IsDebuggerPresent())
					{
						double HangDuration = RenderThreadTimeoutClock.Seconds() - StartTime;
						// TODO: Walk the wait chain instead of explicitly setting the render thread as the hung thread id
						PRAGMA_DISABLE_DEPRECATION_WARNINGS
							uint32 ThreadThatHung = GRenderThreadId;
						PRAGMA_ENABLE_DEPRECATION_WARNINGS
							HandleRenderTaskHang(ThreadThatHung, HangDuration);
					}
				}
#endif
#endif
			} while (!bDone);

			NumRecursiveCalls--;
		}
	}
}


static struct FRenderCommandFenceBundlerState
{
	TOptional<UE::Tasks::FTaskEvent> Event;
	FRenderCommandPipeBitArray RenderCommandPipeBits;
	int32 RecursionDepth = 0;

} GRenderCommandFenceBundlerState;

#define UE_RENDER_COMMAND_FENCE_BUNDLER_REGION TEXT("Render Command Fence Bundler")

void StartRenderCommandFenceBundler()
{
	if (!GIsThreadedRendering || !GRenderCommandFenceBundling)
	{
		return;
	}

	check(IsInGameThread());
	check(!GRenderCommandFenceBundlerState.Event.IsSet() == !GRenderCommandFenceBundlerState.RecursionDepth);

	++GRenderCommandFenceBundlerState.RecursionDepth;

	if (GRenderCommandFenceBundlerState.RecursionDepth > 1)
	{
		return;
	}

	GRenderCommandFenceBundlerState.Event.Emplace(TEXT("RenderCommandFenceBundlerEvent"));

	// Stop render command pipes so that the bundled render command fence is serialized with other render commands.
	GRenderCommandFenceBundlerState.RenderCommandPipeBits = UE::RenderCommandPipe::StopRecording();

	StartBatchedRelease();

	UE_RENDER_COMMAND_BEGIN_REGION(UE_RENDER_COMMAND_FENCE_BUNDLER_REGION);
}

void FlushRenderCommandFenceBundler()
{
	if (GRenderCommandFenceBundlerState.Event)
	{
		EndBatchedRelease();

		ENQUEUE_RENDER_COMMAND(InsertFence)(
			[CompletionEvent = MoveTemp(*GRenderCommandFenceBundlerState.Event)](FRHICommandListBase&) mutable
			{
				CompletionEvent.Trigger();
			});

		GRenderCommandFenceBundlerState.Event.Emplace(TEXT("RenderCommandFenceBundlerEvent"));

		StartBatchedRelease();
	}
}

void StopRenderCommandFenceBundler()
{
	if (!GIsThreadedRendering || !GRenderCommandFenceBundlerState.Event)
	{
		return;
	}

	TOptional<UE::Tasks::FTaskEvent>& CompletionEvent = GRenderCommandFenceBundlerState.Event;

	check(CompletionEvent);
	check(!CompletionEvent->IsCompleted());
	check(GRenderCommandFenceBundlerState.RecursionDepth > 0);

	--GRenderCommandFenceBundlerState.RecursionDepth;

	if (GRenderCommandFenceBundlerState.RecursionDepth > 0)
	{
		return;
	}

	UE_RENDER_COMMAND_END_REGION(UE_RENDER_COMMAND_FENCE_BUNDLER_REGION);

	EndBatchedRelease();

	ENQUEUE_RENDER_COMMAND(InsertFence)(
		[CompletionEvent = MoveTemp(*CompletionEvent)](FRHICommandListBase&) mutable
		{
			CompletionEvent.Trigger();
		});

	CompletionEvent.Reset();

	// Restart render command pipes that were previously recording.
	UE::RenderCommandPipe::StartRecording(GRenderCommandFenceBundlerState.RenderCommandPipeBits);
	GRenderCommandFenceBundlerState.RenderCommandPipeBits.Empty();
}

FRenderCommandFence::FRenderCommandFence() = default;
FRenderCommandFence::~FRenderCommandFence() = default;

void FRenderCommandFence::BeginFence(ESyncDepth SyncDepth)
{
	if (!GIsThreadedRendering)
	{
		return;
	}

	check(IsInGameThread());

	if (GRenderCommandFenceBundlerState.Event && SyncDepth == ESyncDepth::RenderThread)
	{
		// Case for game->render thread syncs when fence bundling is enabled. These are used
		// throughout the engine when resources are destroyed. The fence bundling is an optimization
		// to avoid the overhead of hundreds of individual fences.
		// We aren't syncing any deeper than the render thread, so just use the bundled fence event.
		CompletionTask = *GRenderCommandFenceBundlerState.Event;
		return;
	}

	TRACE_CPUPROFILER_EVENT_SCOPE(FRenderCommandFence::BeginFence);
	UE::Tasks::FTaskEvent Event{ UE_SOURCE_LOCATION };

	if (GRenderCommandFenceBundlerState.Event)
	{
		// Render command fences are bundled, but we're syncing deeper than the render thread.
		// Flush the fence bundler so we can insert an RHIThread (or deeper) fence in the right location.
		Event.AddPrerequisites(*GRenderCommandFenceBundlerState.Event);
		FlushRenderCommandFenceBundler();
	}

	if (GRenderCommandPipeMode == ERenderCommandPipeMode::All)
	{
		for (FRenderCommandPipe* Pipe : UE::RenderCommandPipe::GetPipes())
		{
			// Skip pipes that aren't recording or replaying any work.
			if (Pipe->IsRecording() && !Pipe->IsEmpty())
			{
				UE::Tasks::FTaskEvent PipeEvent{ UE_SOURCE_LOCATION };
				Event.AddPrerequisites(PipeEvent);

				ENQUEUE_RENDER_COMMAND(BeginFence)(Pipe, [PipeEvent = MoveTemp(PipeEvent)](FRHICommandList&) mutable
					{
						PipeEvent.Trigger();
					});
			}
		}
	}

	ENQUEUE_RENDER_COMMAND(BeginFence)([Event, SyncDepth](FRHICommandListImmediate& RHICmdList) mutable
		{
			if (SyncDepth == ESyncDepth::Swapchain)
			{
				UE::Tasks::FTaskEvent SwapchainEvent{ UE_SOURCE_LOCATION };
				Event.AddPrerequisites(SwapchainEvent);

				RHICmdList.EnqueueLambda([SyncDepth, SwapchainEvent](FRHICommandListImmediate&) mutable
					{
						// This command runs *after* a present has happened, so the counter has already been incremented.
						// Subtracting 1 gives us the index of the frame that has *just* been presented.
						RHITriggerTaskEventOnFlip(GRHIPresentCounter - 1, SwapchainEvent);
					});

				RHICmdList.ImmediateFlush(EImmediateFlushType::DispatchToRHIThread);
			}
			else if (SyncDepth == ESyncDepth::RHIThread)
			{
				Event.AddPrerequisites(GRHICommandList.Submit({}, ERHISubmitFlags::SubmitToGPU));
			}

			TRACE_CPUPROFILER_EVENT_SCOPE(SyncTrigger_RenderThread);
			Event.Trigger();
		});

	CompletionTask = MoveTemp(Event);
}

/**
 * Waits for pending fence commands to retire.
 */
void FRenderCommandFence::Wait(bool bProcessGameThreadTasks) const
{
	if (!IsFenceComplete())
	{
		FRenderCommandList::FFlushScope FlushScope;
		FlushRenderCommandFenceBundler();
		GameThreadWaitForTask(CompletionTask, bProcessGameThreadTasks);
		CompletionTask = {}; // release the internal memory as soon as it's not needed anymore
	}
}

bool FRenderCommandFence::IsFenceComplete() const
{
	if (!GIsThreadedRendering)
	{
		return true;
	}
	check(IsInGameThread() || IsInAsyncLoadingThread());
	CheckRenderingThreadHealth();
	if (CompletionTask.IsCompleted())
	{
		CompletionTask = {}; // this frees the handle for other uses, the NULL state is considered completed
		return true;
	}
	return false;
}
