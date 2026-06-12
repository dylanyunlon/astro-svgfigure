// Copyright Epic Games, Inc. All Rights Reserved.

#include "RenderingThreadHeartbeat.h"
#include "HAL/PlatformProcess.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "RenderingThread.h"
#include "Templates/Atomic.h"
#include "TickableObjectRenderThread.h"

/** If the rendering thread is in its idle loop (which ticks rendering tickables) */
TAtomic<bool> GRunRenderingThreadHeartbeat;

TAtomic<int32> GOutstandingHeartbeats = 0;

/** rendering tickables shouldn't be updated during a flush */
TAtomic<int32> GSuspendRenderingTickables;

/** Maximum rate the rendering thread will tick tickables when idle (in Hz) */
float GRenderingThreadMaxIdleTickFrequency = 40.f;

FSuspendRenderingTickables::FSuspendRenderingTickables()
{
	++GSuspendRenderingTickables;
}

FSuspendRenderingTickables::~FSuspendRenderingTickables()
{
	--GSuspendRenderingTickables;
}

/** The rendering thread heartbeat runnable object. */
class FRenderingThreadTickHeartbeat : public FRunnable
{
public:
	FRenderingThreadTickHeartbeat(uint32 ThreadNumber)
	{
		GRunRenderingThreadHeartbeat = true;

		UE::Trace::ThreadGroupBegin(TEXT("Render"));
		Thread = FRunnableThread::Create(
			this,
			*FString::Printf(TEXT("RTHeartBeat %d"), ThreadNumber),
			80 * 1024,
			TPri_AboveNormal,
			FPlatformAffinity::GetRTHeartBeatMask()
		);
		UE::Trace::ThreadGroupEnd();
	}

	~FRenderingThreadTickHeartbeat()
	{
		// Signal to stop running the thread.
		GRunRenderingThreadHeartbeat = false;

		// Wait for the rendering thread heartbeat to return.
		Thread->WaitForCompletion();

		delete Thread;
	}

	// FRunnable interface.
	virtual bool Init() final
	{
		GSuspendRenderingTickables = 0;
		GOutstandingHeartbeats.Store(0);
		return true;
	}

	virtual uint32 Run() final
	{
		while (GRunRenderingThreadHeartbeat.Load(EMemoryOrder::Relaxed))
		{
			FPlatformProcess::Sleep(1.f / (4.0f * GRenderingThreadMaxIdleTickFrequency));
			if (GOutstandingHeartbeats.Load() < 4)
			{
				++GOutstandingHeartbeats;

				ENQUEUE_RENDER_COMMAND(HeartbeatTickTickables)(
					[](FRHICommandListImmediate& RHICmdList)
					{
						--GOutstandingHeartbeats;

						// make sure that rendering thread tickables get a chance to tick, even if the render thread is starving
						// but if GSuspendRenderingTickables is != 0 a flush is happening so don't tick during it
						if (!GSuspendRenderingTickables.Load(EMemoryOrder::Relaxed))
						{
							TickRenderingTickables(RHICmdList);
						}
					});
			}
		}
		return 0;
	}

private:
	FRunnableThread* Thread = nullptr;
};

static FRenderingThreadTickHeartbeat* GRenderingThreadRunnableHeartbeat = nullptr;

void StartRenderingThreadHeartbeatThread(uint32 ThreadNumber)
{
	GRenderingThreadRunnableHeartbeat = new FRenderingThreadTickHeartbeat(ThreadNumber);
}

void StopRenderingThreadHeartbeatThread()
{
	if (GRunRenderingThreadHeartbeat)
	{
		GRenderingThreadRunnableHeartbeat->Stop();

		delete GRenderingThreadRunnableHeartbeat;
		GRenderingThreadRunnableHeartbeat = nullptr;
	}
}
