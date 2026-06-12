// Copyright Epic Games, Inc. All Rights Reserved.

#include "RenderingThread.h"
#include "DataDrivenShaderPlatformInfo.h"
#include "HAL/ExceptionHandling.h" // IWYU pragma: keep
#include "HAL/LowLevelMemTracker.h"
#include "HAL/PlatformApplicationMisc.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "HAL/ThreadHeartBeat.h"
#include "HAL/ThreadManager.h"
#include "Misc/App.h"
#include "Misc/CommandLine.h"
#include "Misc/CoreDelegates.h"
#include "Misc/CoreStats.h"
#include "Misc/OutputDeviceRedirector.h"
#include "Misc/ScopeLock.h"
#include "Misc/TimeGuard.h"
#include "ProfilingDebugging/CountersTrace.h"
#include "ProfilingDebugging/CsvProfiler.h"
#include "ProfilingDebugging/MiscTrace.h"
#include "RenderCommandFence.h"
#include "RenderCommandStateStream.h"
#include "RenderCore.h"
#include "RenderDeferredCleanup.h"
#include "RenderResource.h"
#include "RenderingThreadHeartbeat.h"
#include "RHI.h"
#include "RHIThread.h"
#include "RHIUtilities.h"
#include "Stats/StatsData.h"
#include "Stats/StatsSystem.h"
#include "TickableObjectRenderThread.h"

//
// Globals

// Begin RenderTimer.h
uint32 GRenderThreadTime = 0;
uint32 GRenderThreadWaitTime = 0;
uint32 GRHIThreadTime = 0;
uint32 GRenderThreadTimeCriticalPath = 0;

uint32 GGameThreadTime = 0;
uint32 GGameThreadWaitTime = 0;
uint32 GGameThreadTimeCriticalPath = 0;
uint32 GSwapBufferTime = 0;
// End RenderTimer.h

FCoreRenderDelegates::FOnFlushRenderingCommandsStart FCoreRenderDelegates::OnFlushRenderingCommandsStart;
FCoreRenderDelegates::FOnFlushRenderingCommandsEnd FCoreRenderDelegates::OnFlushRenderingCommandsEnd;

UE_TRACE_CHANNEL_DEFINE(RenderCommandsChannel, "Render thread command execution and synchronization.");

bool GIsThreadedRendering = false;
bool GUseThreadedRendering = false;
TOptional<bool> GPendingUseThreadedRendering;

#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
	TAtomic<bool> GMainThreadBlockedOnRenderThread(false);
#endif // #if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)

/** If the rendering thread has been terminated by an unhandled exception, this contains the error message. */
FString GRenderingThreadError;

/**
 * Polled by the game thread to detect crashes in the rendering thread.
 * If the rendering thread crashes, it sets this variable to false.
 */
volatile bool GIsRenderingThreadHealthy = true;

/**
 * RT Task Graph polling.
 */
extern CORE_API bool GRenderThreadPollingOn;
extern CORE_API int32 GRenderThreadPollPeriodMs;

static void OnRenderThreadPollPeriodMsChanged(IConsoleVariable* Var)
{
	const int32 DesiredRTPollPeriod = Var->GetInt();

	GRenderThreadPollingOn = (DesiredRTPollPeriod >= 0);
	ENQUEUE_RENDER_COMMAND(WakeupCommand)([DesiredRTPollPeriod](FRHICommandListImmediate&)
	{
		GRenderThreadPollPeriodMs = DesiredRTPollPeriod;
	});
}

static FAutoConsoleVariable CVarRenderThreadPollPeriodMs(
	TEXT("TaskGraph.RenderThreadPollPeriodMs"),
	1,
	TEXT("Render thread polling period in milliseconds. If value < 0, task graph tasks explicitly wake up RT, otherwise RT polls for tasks."),
	FConsoleVariableDelegate::CreateStatic(&OnRenderThreadPollPeriodMsChanged)
);

inline ERenderCommandPipeMode GetValidatedRenderCommandPipeMode(int32 CVarValue)
{
	ERenderCommandPipeMode Mode = ERenderCommandPipeMode::None;

	switch (CVarValue)
	{
	case 1:
		Mode = ERenderCommandPipeMode::RenderThread;
		break;
	case 2:
		Mode = ERenderCommandPipeMode::All;
		break;
	}

	const bool bAllowThreading = !GRHICommandList.Bypass() && FApp::ShouldUseThreadingForPerformance() && GIsThreadedRendering;

	if (Mode == ERenderCommandPipeMode::All && !bAllowThreading)
	{
		Mode = ERenderCommandPipeMode::RenderThread;
	}

	if (!FApp::CanEverRender() || IsMobilePlatform(GMaxRHIShaderPlatform))
	{
		Mode = ERenderCommandPipeMode::None;
	}

	return Mode;
}

ERenderCommandPipeMode GRenderCommandPipeMode = ERenderCommandPipeMode::None;
FAutoConsoleVariable CVarRenderCommandPipeMode(
	TEXT("r.RenderCommandPipeMode"),
	2,
	TEXT("Controls behavior of the main render thread command pipe.")
	TEXT(" 0: Render commands are launched individually as tasks;\n")
	TEXT(" 1: Render commands are enqueued into a render command pipe for the render thread only.;\n")
	TEXT(" 2: Render commands are enqueued into a render command pipe for all declared pipes.;\n"),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* Variable)
	{
		UE::RenderCommandPipe::StopRecording();
		GRenderCommandPipeMode = GetValidatedRenderCommandPipeMode(Variable->GetInt());
	}));

/** The rendering thread main loop */
void RenderingThreadMain(FEvent* TaskGraphBoundSyncEvent)
{
	LLM_SCOPE(ELLMTag::RenderingThreadMemory);

	ENamedThreads::Type RenderThread = ENamedThreads::Type(ENamedThreads::ActualRenderingThread);

	ENamedThreads::SetRenderThread(RenderThread);
	ENamedThreads::SetRenderThread_Local(ENamedThreads::Type(ENamedThreads::ActualRenderingThread_Local));

	FTaskGraphInterface::Get().AttachToThread(RenderThread);
	FPlatformMisc::MemoryBarrier();

	// Inform main thread that the render thread has been attached to the taskgraph and is ready to receive tasks
	if (TaskGraphBoundSyncEvent)
	{
		TaskGraphBoundSyncEvent->Trigger();
	}

#if STATS
	if (FThreadStats::WillEverCollectData())
	{
		FTaskTagScope Scope(ETaskTag::ERenderingThread);
		FThreadStats::ExplicitFlush(); // flush the stats and set update the scope so we don't flush again until a frame update, this helps prevent fragmentation
	}
#endif

	FCoreDelegates::PostRenderingThreadCreated.Broadcast();
	check(GIsThreadedRendering);
	{
		FTaskTagScope TaskTagScope(ETaskTag::ERenderingThread);

		// Acquire rendering context ownership on the current thread, unless using an RHI thread, which will be the real owner
		FScopedRHIThreadOwnership ThreadOwnershipScope(!IsRunningRHIInSeparateThread());

		FTaskGraphInterface::Get().ProcessThreadUntilRequestReturn(RenderThread);
	}
	FPlatformMisc::MemoryBarrier();

	check(!GIsThreadedRendering);

	FCoreDelegates::PreRenderingThreadDestroyed.Broadcast();

#if STATS
	if (FThreadStats::WillEverCollectData())
	{
		FThreadStats::ExplicitFlush(); // Another explicit flush to clean up the ScopeCount established above for any stats lingering since the last frame
	}
#endif

	ENamedThreads::SetRenderThread(ENamedThreads::GameThread);
	ENamedThreads::SetRenderThread_Local(ENamedThreads::GameThread_Local);
	FPlatformMisc::MemoryBarrier();
}

/**
 * Advances stats for the rendering thread.
 */
#if STATS
static void AdvanceRenderingThreadStats(int64 StatsFrame, int32 DisableChangeTagStartFrame)
{
	int64 Frame = StatsFrame;
	if (!FThreadStats::IsCollectingData() || DisableChangeTagStartFrame != FThreadStats::PrimaryDisableChangeTag())
	{
		Frame = -StatsFrame; // mark this as a bad frame
	}
	FThreadStats::AddMessage(FStatConstants::AdvanceFrame.GetEncodedName(), EStatOperation::AdvanceFrameEventRenderThread, Frame);

	// Propagate the stats frame value down to the end-of-pipe thread.
	UE::Stats::FStats::StatsFrameRT = Frame;

	if (IsInActualRenderingThread())
	{
		FThreadStats::ExplicitFlush();
	}
}
#endif

/**
 * Advances stats for the rendering thread. Called from the game thread.
 */
void AdvanceRenderingThreadStatsGT(bool bDiscardCallstack, int64 StatsFrame, int32 DisableChangeTagStartFrame)
{
#if STATS
	ENQUEUE_RENDER_COMMAND(RenderingThreadTickCommand)(
		[StatsFrame, DisableChangeTagStartFrame](FRHICommandList& RHICmdList)
		{
			AdvanceRenderingThreadStats(StatsFrame, DisableChangeTagStartFrame);
		}
	);
#endif
	if (bDiscardCallstack)
	{
		// we need to flush the rendering thread here, otherwise it can get behind and then the stats will get behind.
		FlushRenderingCommands();
	}
}

/** The rendering thread runnable object. */
class FRenderingThread : public FRunnable
{
private:
	// Sync event to make sure that render thread is bound to the task graph before main thread queues work against it.
	FEvent* TaskGraphBoundSyncEvent;

	FRunnableThread* Thread;

public:
	FRenderingThread() = delete;
	FRenderingThread(uint32 ThreadNumber)
	{
		TaskGraphBoundSyncEvent = FPlatformProcess::GetSynchEventFromPool(true);

		UE::Trace::ThreadGroupBegin(TEXT("Render"));
		Thread = FRunnableThread::Create(
			this,
			*FString::Printf(TEXT("RenderThread %u"), ThreadNumber),
			0,
			FPlatformAffinity::GetRenderingThreadPriority(),
			FPlatformAffinity::GetRenderingThreadMask(),
			FPlatformAffinity::GetRenderingThreadFlags()
		);
		UE::Trace::ThreadGroupEnd();

		// Wait for the render thread to attach to the task graph before dispatching any work.
		TaskGraphBoundSyncEvent->Wait();
	}

	virtual ~FRenderingThread()
	{
		// Wait for the rendering thread to fully exit before destroying.
		Thread->WaitForCompletion();
		delete Thread;

		FPlatformProcess::ReturnSynchEventToPool(TaskGraphBoundSyncEvent);
		TaskGraphBoundSyncEvent = nullptr;
	}

	virtual bool Init() final
	{
PRAGMA_DISABLE_DEPRECATION_WARNINGS
		GRenderThreadId = FPlatformTLS::GetCurrentThreadId();
PRAGMA_ENABLE_DEPRECATION_WARNINGS
		FTaskTagScope::SetTagNone();
		return true;
	}

	virtual void Exit() final
	{
PRAGMA_DISABLE_DEPRECATION_WARNINGS
		GRenderThreadId = 0;
PRAGMA_ENABLE_DEPRECATION_WARNINGS
	}

#if PLATFORM_WINDOWS && !PLATFORM_SEH_EXCEPTIONS_DISABLED
	static int32 FlushRHILogsAndReportCrash(Windows::LPEXCEPTION_POINTERS ExceptionInfo)
	{
		if (GDynamicRHI)
		{
			GDynamicRHI->FlushPendingLogs();
		}

		return ReportCrash(ExceptionInfo);
	}
#endif

	void SetupRenderThread()
	{
		FTaskTagScope Scope(ETaskTag::ERenderingThread);
		FPlatformProcess::SetupRenderThread();
	}

	virtual uint32 Run() final
	{
		FMemory::SetupTLSCachesOnCurrentThread();
		SetupRenderThread();

#if PLATFORM_WINDOWS
		const bool bNoExceptionHandler = FParse::Param(FCommandLine::Get(), TEXT("noexceptionhandler"));
		if (!bNoExceptionHandler && (!FPlatformMisc::IsDebuggerPresent() || GAlwaysReportCrash))
		{
#if !PLATFORM_SEH_EXCEPTIONS_DISABLED
			__try
#endif
			{
				RenderingThreadMain(TaskGraphBoundSyncEvent);
			}
#if !PLATFORM_SEH_EXCEPTIONS_DISABLED
			__except (FPlatformMisc::GetCrashHandlingType() == ECrashHandlingType::Default ?
				FlushRHILogsAndReportCrash(GetExceptionInformation()) :
				EXCEPTION_CONTINUE_SEARCH)
			{
#if !NO_LOGGING
				// Dump the error and flush the log. This is the same logging behavior as FWindowsErrorOutputDevice::HandleError which is called in GuardedMain's caller's __except
				FDebug::LogFormattedMessageWithCallstack(LogWindows.GetCategoryName(), __FILE__, __LINE__, TEXT("=== Critical error: ==="), GErrorHist, ELogVerbosity::Error);
#endif
				GLog->Panic();

				GRenderingThreadError = GErrorHist;

				// Use a memory barrier to ensure that the game thread sees the write to GRenderingThreadError before
				// the write to GIsRenderingThreadHealthy.
				FPlatformMisc::MemoryBarrier();

				GIsRenderingThreadHealthy = false;
			}
#endif
		}
		else
#endif // PLATFORM_WINDOWS
		{
			RenderingThreadMain(TaskGraphBoundSyncEvent);
		}

		FMemory::ClearAndDisableTLSCachesOnCurrentThread();
		return 0;
	}
};

static FRenderingThread* GRenderingThread = nullptr;

// not done in the CVar system as we don't access to render thread specifics there
struct FConsoleRenderThreadPropagation : public IConsoleThreadPropagation
{
	virtual void OnCVarChange(int32& Dest, int32 NewValue)
	{
		int32* DestPtr = &Dest;
		ENQUEUE_RENDER_COMMAND(OnCVarChange1)(
			[DestPtr, NewValue](FRHICommandListImmediate& RHICmdList)
			{
				*DestPtr = NewValue;
			});
	}
	
	virtual void OnCVarChange(float& Dest, float NewValue)
	{
		float* DestPtr = &Dest;
		ENQUEUE_RENDER_COMMAND(OnCVarChange2)(
			[DestPtr, NewValue](FRHICommandListImmediate& RHICmdList)
			{
				*DestPtr = NewValue;
			});
	}

	virtual void OnCVarChange(bool& Dest, bool NewValue)
	{
		bool* DestPtr = &Dest;
		ENQUEUE_RENDER_COMMAND(OnCVarChange2)(
			[DestPtr, NewValue](FRHICommandListImmediate& RHICmdList)
			{
				*DestPtr = NewValue;
			});
	}
	
	virtual void OnCVarChange(FString& Dest, const FString& NewValue)
	{
		FString* DestPtr = &Dest;
		ENQUEUE_RENDER_COMMAND(OnCVarChange3)(
			[DestPtr, NewValue](FRHICommandListImmediate& RHICmdList)
			{
				*DestPtr = NewValue;
			});
	}

	virtual void OnCVarChange(FName& Dest, const FName& NewValue)
	{
		FName* DestPtr = &Dest;
		ENQUEUE_RENDER_COMMAND(OnCVarChange3)(
			[DestPtr, NewValue](FRHICommandListImmediate& RHICmdList)
			{
				*DestPtr = NewValue;
			});
	}

	static FConsoleRenderThreadPropagation& GetSingleton()
	{
		static FConsoleRenderThreadPropagation This;

		return This;
	}
};

static void StartRenderingThread(ERHIThreadMode RHIThreadMode)
{
	check(IsInGameThread());

	// Do nothing if we're already in the right mode
	if (GIsThreadedRendering || !GUseThreadedRendering)
	{
		check(GIsThreadedRendering == GUseThreadedRendering);
		return;
	}

	check(!IsRHIThreadRunning() && !GIsRunningRHIInSeparateThread_InternalUseOnly && !GIsRunningRHIInDedicatedThread_InternalUseOnly && !GIsRunningRHIInTaskThread_InternalUseOnly);

	// Pause asset streaming to prevent rendercommands from being enqueued.
	SuspendTextureStreamingRenderTasks();

	// Flush GT since render commands issued by threads other than GT are sent to
	// the main queue of GT when RT is disabled. Without this flush, those commands
	// will run on GT after RT is enabled
	FlushRenderingCommands();

	GDynamicRHI->RHIReleaseThreadOwnership();

	switch (GRHISupportsRHIThread ? RHIThreadMode : ERHIThreadMode::None)
	{
	case ERHIThreadMode::DedicatedThread:
		GIsRunningRHIInSeparateThread_InternalUseOnly  = true;
		GIsRunningRHIInDedicatedThread_InternalUseOnly = true;
		GIsRunningRHIInTaskThread_InternalUseOnly      = false;

		// Start the dedicated RHI thread
		StartRHIThread();
		break;

	case ERHIThreadMode::Tasks:
		GIsRunningRHIInSeparateThread_InternalUseOnly  = true;
		GIsRunningRHIInDedicatedThread_InternalUseOnly = false;
		GIsRunningRHIInTaskThread_InternalUseOnly      = true;
		break;

	default:
		checkNoEntry();
		[[fallthrough]];

	case ERHIThreadMode::None:
		GIsRunningRHIInSeparateThread_InternalUseOnly  = false;
		GIsRunningRHIInDedicatedThread_InternalUseOnly = false;
		GIsRunningRHIInTaskThread_InternalUseOnly      = false;
		break;
	}

	// Turn on the threaded rendering flag.
	GIsThreadedRendering = true;

	static uint32 ThreadCount = 0;

	// Create the rendering thread. The constructor creates the OS thread and waits for
	// it to attach to the task graph before returning.
	GRenderingThread = new FRenderingThread(ThreadCount);

	// register
	IConsoleManager::Get().RegisterThreadPropagation(0, &FConsoleRenderThreadPropagation::GetSingleton());

	ENQUEUE_RENDER_COMMAND(LatchBypass)([](FRHICommandListImmediate&)
	{
		GRHICommandList.LatchBypass();
	});

	// ensure the thread has actually started and is idling
	FRenderCommandFence Fence;
	Fence.BeginFence();
	Fence.Wait();

	GRenderCommandPipeMode = GetValidatedRenderCommandPipeMode(CVarRenderCommandPipeMode->GetInt());

	// Create the rendering thread heartbeat thread
	StartRenderingThreadHeartbeatThread(ThreadCount);

	ThreadCount++;

	// Update can now resume.
	ResumeTextureStreamingRenderTasks();
}

static FStopRenderingThread GStopRenderingThreadDelegate;

FDelegateHandle RegisterStopRenderingThreadDelegate(const FStopRenderingThread::FDelegate& InDelegate)
{
	return GStopRenderingThreadDelegate.Add(InDelegate);
}

void UnregisterStopRenderingThreadDelegate(FDelegateHandle InDelegateHandle)
{
	GStopRenderingThreadDelegate.Remove(InDelegateHandle);
}

static void StopRenderingThread()
{
	// This function is not thread-safe. Ensure it is only called by the main game thread.
	check(IsInGameThread());

	if (!GIsThreadedRendering)
	{
		return;
	}

	// unregister
	IConsoleManager::Get().RegisterThreadPropagation();

	// stop the render thread heartbeat first
	StopRenderingThreadHeartbeatThread();

	GStopRenderingThreadDelegate.Broadcast();

	// Get the list of objects which need to be cleaned up when the rendering thread is done with them.
	FPendingCleanupObjects* PendingCleanupObjects = GetPendingCleanupObjects();

	// Make sure we're not in the middle of streaming textures.
	SuspendTextureStreamingRenderTasks();

	// Wait for the rendering thread to finish executing all enqueued commands.
	FlushRenderingCommands();

	// Shutdown RHI thread
	StopRHIThread();

	GIsRunningRHIInSeparateThread_InternalUseOnly  = false;
	GIsRunningRHIInDedicatedThread_InternalUseOnly = false;
	GIsRunningRHIInTaskThread_InternalUseOnly      = false;

	// Turn off the threaded rendering flag.
	GIsThreadedRendering = false;

	{
		FGraphEventRef QuitTask = TGraphTask<FReturnGraphTask>::CreateTask(nullptr, ENamedThreads::GameThread).ConstructAndDispatchWhenReady(ENamedThreads::GetRenderThread());

		// Busy wait while BP debugging, to avoid opportunistic execution of game thread tasks
		// If the game thread is already executing tasks, then we have no choice but to spin
		if (GIntraFrameDebuggingGameThread || FTaskGraphInterface::Get().IsThreadProcessingTasks(ENamedThreads::GameThread))
		{
			while ((QuitTask.GetReference() != nullptr) && !QuitTask->IsComplete())
			{
				FPlatformProcess::Sleep(0.0f);
			}
		}
		else
		{
			QUICK_SCOPE_CYCLE_COUNTER(STAT_StopRenderingThread);
			FTaskGraphInterface::Get().WaitUntilTaskCompletes(QuitTask, ENamedThreads::GameThread_Local);
		}
	}

	// Destroying FRenderingThread waits for the OS thread to exit before returning.
	delete GRenderingThread;
	GRenderingThread = nullptr;

	// Make sure the game thread is marked as the thread for rendering.
	GDynamicRHI->RHIAcquireThreadOwnership();

	// Make sure bypass is set correctly with the render thread offline.
	GRHICommandList.LatchBypass();

	// Delete the pending cleanup objects which were in use by the rendering thread.
	delete PendingCleanupObjects;

	// Update can now resume with render thread being the game thread.
	ResumeTextureStreamingRenderTasks();

	check(!IsRHIThreadRunning());
}

static ERHIThreadMode GRHIThreadMode = ERHIThreadMode::DedicatedThread;

void LatchRenderThreadConfiguration()
{
	check(IsInGameThread());

	// Check for pending state changes from the "togglerenderingthread" and "r.RHIThread.Enable" commands.
	if ((GPendingUseThreadedRendering.IsSet() && GPendingUseThreadedRendering != GUseThreadedRendering) ||
		(GPendingRHIThreadMode.IsSet() && *GPendingRHIThreadMode != GRHIThreadMode))
	{
		// Something changed. Stop and restart the rendering and RHI threads according to the new config.
		StopRenderingThread();

		if (GPendingUseThreadedRendering.IsSet())
		{
			GUseThreadedRendering = *GPendingUseThreadedRendering;
			GPendingUseThreadedRendering.Reset();
		}

		if (GPendingRHIThreadMode.IsSet())
		{
			GRHIThreadMode = *GPendingRHIThreadMode;
			GPendingRHIThreadMode.Reset();
		}

		StartRenderingThread(GRHIThreadMode);
	}

	ENQUEUE_RENDER_COMMAND(LatchBypass)([](FRHICommandListImmediate&)
	{
		GRHICommandList.LatchBypass();
	});
}

void InitRenderingThread()
{
	UE_CALL_ONCE([]()
	{
		if (FParse::Param(FCommandLine::Get(), TEXT("norhithread")))
		{
			GRHIThreadMode = ERHIThreadMode::None;
		}

		SCOPED_BOOT_TIMING("StartRenderingThread");
		StartRenderingThread(GRHIThreadMode);
	});
}

void ShutdownRenderingThread()
{
	UE_CALL_ONCE([]()
	{
		StopRenderingThread();
	});
}

bool IsRenderingThreadHealthy()
{
	return GIsRenderingThreadHealthy;
}

void CheckRenderingThreadHealth()
{
	if (!IsRenderingThreadHealthy())
	{
		GErrorHist[0] = 0;
		GIsCriticalError = false;
		UE_LOGF(LogRendererCore, Fatal,"Rendering thread exception:\r\n%ls",*GRenderingThreadError);
	}

	if (IsInGameThread())
	{
		if (!GIsCriticalError)
		{
			GLog->FlushThreadedLogs(EOutputDeviceRedirectorFlushOptions::Async);
		}
#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
		TGuardValue<TAtomic<bool>, bool> GuardMainThreadBlockedOnRenderThread(GMainThreadBlockedOnRenderThread,true);
#endif
		//QUICK_SCOPE_CYCLE_COUNTER(STAT_PumpMessages);
		FPlatformApplicationMisc::PumpMessages(false);
	}
}

/**
 * Waits for the rendering thread to finish executing all pending rendering commands.  Should only be used from the game thread.
 */
void FlushRenderingCommands()
{
	if (!GIsRHIInitialized)
	{
		return;
	}

	TRACE_CPUPROFILER_EVENT_SCOPE(FlushRenderingCommands);
	FRenderCommandList::FFlushScope FlushScope;
	FCoreRenderDelegates::OnFlushRenderingCommandsStart.Broadcast();
	FSuspendRenderingTickables SuspendRenderingTickables;

	// Need to flush GT because render commands from threads other than GT are sent to
	// the main queue of GT when RT is disabled
	if (!GIsThreadedRendering
		&& !FTaskGraphInterface::Get().IsThreadProcessingTasks(ENamedThreads::GameThread)
		&& !FTaskGraphInterface::Get().IsThreadProcessingTasks(ENamedThreads::GameThread_Local))
	{
		FTaskGraphInterface::Get().ProcessThreadUntilIdle(ENamedThreads::GameThread);
		FTaskGraphInterface::Get().ProcessThreadUntilIdle(ENamedThreads::GameThread_Local);
	}

	UE::RenderCommandPipe::StopRecording();

	ENQUEUE_RENDER_COMMAND(FlushPendingDeleteRHIResourcesCmd)([](FRHICommandListImmediate& RHICmdList)
	{
		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThreadFlushResources);
		//double flush to flush out the deferred deletions queued into the ImmediateCmdList
		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
	});

	// Find the objects which may be cleaned up once the rendering thread command queue has been flushed.
	FPendingCleanupObjects* PendingCleanupObjects = GetPendingCleanupObjects();

	// Issue a fence command to the rendering thread and wait for it to complete.
	// Use the frame end sync here, so that it cleans up outstanding graph events, which is necessary on engine shutdown.
	FFrameEndSync::Sync(FFrameEndSync::EFlushMode::Threads);

	// Delete the objects which were enqueued for deferred cleanup before the command queue flush.
	delete PendingCleanupObjects;

	FCoreRenderDelegates::OnFlushRenderingCommandsEnd.Broadcast();
}

FRHICommandListImmediate& GetImmediateCommandList_ForRenderCommand()
{
	return FRHICommandListExecutor::GetImmediateCommandList();
}
