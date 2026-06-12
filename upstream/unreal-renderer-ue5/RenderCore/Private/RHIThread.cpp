// Copyright Epic Games, Inc. All Rights Reserved.

#include "RHIThread.h"
#include "CoreGlobals.h"
#include "HAL/IConsoleManager.h"
#include "HAL/PlatformAffinity.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "ProfilingDebugging/CsvProfiler.h"
#include "RenderingThread.h"

TOptional<ERHIThreadMode> GPendingRHIThreadMode;

static void HandleRHIThreadEnableChanged(const TArray<FString>& Args)
{
	check(IsInGameThread());
	switch (Args.Num() == 1 ? FCString::Atoi(*Args[0]) : -1)
	{
	case 0:
		GPendingRHIThreadMode = ERHIThreadMode::None;
		UE_LOGF(LogConsoleResponse, Display, "RHI thread will be disabled.");
		break;

	case 1:
		GPendingRHIThreadMode = ERHIThreadMode::DedicatedThread;
		UE_LOGF(LogConsoleResponse, Display, "RHI thread will be enabled (dedicated thread).");
		break;

	case 2:
		GPendingRHIThreadMode = ERHIThreadMode::Tasks;
		UE_LOGF(LogConsoleResponse, Display, "RHI thread will be enabled (task threads).");
		break;

	default:
		UE_LOGF(LogConsoleResponse, Display, "Usage: r.RHIThread.Enable 0=off,  1=dedicated thread,  2=task threads; Currently %d", IsRunningRHIInSeparateThread() ? (IsRunningRHIInDedicatedThread() ? 1 : 2) : 0);
		break;
	}
}

static FAutoConsoleCommand CVarRHIThreadEnable(
	TEXT("r.RHIThread.Enable"),
	TEXT("Enables/disabled the RHI Thread and determine if the RHI work runs on a dedicated thread or not.\n"),
	FConsoleCommandWithArgsDelegate::CreateStatic(&HandleRHIThreadEnableChanged)
);

/** The RHI thread runnable object. */
class FRHIThread : private FRunnable
{
	FRunnableThread* Thread = nullptr;

public:
	FRHIThread()
	{
		check(IsInGameThread());

		UE::Trace::ThreadGroupBegin(TEXT("Render"));
		Thread = FRunnableThread::Create(
			this,
			TEXT("RHIThread"),
			512 * 1024,
			FPlatformAffinity::GetRHIThreadPriority(),
			FPlatformAffinity::GetRHIThreadMask(),
			FPlatformAffinity::GetRHIThreadFlags()
		);
		UE::Trace::ThreadGroupEnd();

		check(Thread);
	}

	~FRHIThread()
	{
		check(IsInGameThread());

		// Signal the task graph to make the RHI thread exit, and wait for it.
		TGraphTask<FReturnGraphTask>::CreateTask(nullptr, ENamedThreads::GameThread).ConstructAndDispatchWhenReady(ENamedThreads::RHIThread);
		Thread->WaitForCompletion();

		delete Thread;
	}

	virtual uint32 Run() override
	{
		LLM_SCOPE(ELLMTag::RHIMisc);

#if CSV_PROFILER_STATS
		FCsvProfiler::Get()->SetRHIThreadId(FPlatformTLS::GetCurrentThreadId());
#endif
		{
			FTaskTagScope Scope(ETaskTag::ERhiThread);

			FMemory::SetupTLSCachesOnCurrentThread();
			{
				FScopedRHIThreadOwnership ThreadOwnershipScope(true);

				FTaskGraphInterface::Get().AttachToThread(ENamedThreads::RHIThread);
				FTaskGraphInterface::Get().ProcessThreadUntilRequestReturn(ENamedThreads::RHIThread);
			}
			FMemory::ClearAndDisableTLSCachesOnCurrentThread();
		}

#if CSV_PROFILER_STATS
		FCsvProfiler::Get()->SetRHIThreadId(0);
#endif

		return 0;
	}
};

static FRHIThread* GRHIThread = nullptr;

void StartRHIThread()
{
	GRHIThread = new FRHIThread();
}

void StopRHIThread()
{
	delete GRHIThread;
	GRHIThread = nullptr;
}
