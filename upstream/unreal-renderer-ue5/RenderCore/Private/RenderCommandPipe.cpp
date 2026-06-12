// Copyright Epic Games, Inc. All Rights Reserved.

#include "RenderingThread.h"
#include "Async/TaskTrace.h"
#include "ProfilingDebugging/MiscTrace.h"

#define UE_RENDER_COMMAND_PIPE_RECORD_REGION TEXT("Render Command Pipe Recording")
#define UE_RENDER_COMMAND_PIPE_SYNC_REGION TEXT("Render Command Pipe Synced")

DECLARE_RENDER_COMMAND_TAG(FRenderCommandTag_ExecuteCommandLists, ExecuteCommandLists);

namespace UE::RenderCommandPipe
{
	static const FRenderCommandTag& GetExecuteCommandListTag()
	{
		DECLARE_RENDER_COMMAND_TAG(FTag, ExecuteCommandList);
		return FTag::Get();
	}

	bool FCommandList::Enqueue(FRenderCommandFunctionVariant&& Function, const FRenderCommandTag& Tag)
	{
		return Enqueue(AllocNoDestruct<FExecuteFunctionCommand>(MoveTemp(Function), Tag));
	}

	bool FCommandList::Enqueue(FCommandList* CommandList)
	{
		return Enqueue(AllocNoDestruct<FExecuteCommandListCommand>(CommandList));
	}

	bool FCommandList::Enqueue(FCommand* Command)
	{
#if DO_CHECK
		check(!bClosed);
#endif

		bool bWasEmpty = IsEmpty();
		if (bWasEmpty)
		{
			Commands.Head = Commands.Tail = Command;
		}
		else
		{
			Commands.Tail->Next = Command;
			Commands.Tail = Command;
		}
		Commands.Num++;
		return bWasEmpty;
	}

	void FCommandList::Release()
	{
		if (!IsEmpty())
		{
			ensureMsgf(false, TEXT("UE::RenderCommandPipe::FCommandList was released without consuming commands. You must call ConsumeCommands first."));

			for (FCommand* Command = Commands.Head; Command; Command = Command->Next)
			{
				Command->~FCommand();
			}
		}
		Commands = {};
	}

} //! UE::RenderCommandPipe


static void ExecuteCommand_NoMarker(FRHICommandList* RHICmdList, const FRenderCommandFunctionVariant& Function)
{
	switch (Function.GetIndex())
	{
	case FRenderCommandFunctionVariant::IndexOfType<TUniqueFunction<void(FRHICommandListImmediate&)>>():
		Function.Get<TUniqueFunction<void(FRHICommandListImmediate&)>>()(RHICmdList->GetAsImmediate());
		break;

	case FRenderCommandFunctionVariant::IndexOfType<TUniqueFunction<void(FRHICommandList&)>>():
		Function.Get<TUniqueFunction<void(FRHICommandList&)>>()(*RHICmdList);
		break;

	case FRenderCommandFunctionVariant::IndexOfType<TUniqueFunction<void()>>():
		Function.Get<TUniqueFunction<void()>>()();
		break;

	default: checkNoEntry();
	}
}

static void ExecuteCommand(FRHICommandList* RHICmdList, const FRenderCommandFunctionVariant& Function, const FRenderCommandTag& Tag)
{
	if (Tag.GetStatId().IsValidStat())
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_USE_ON_CHANNEL(Tag.GetSpecId(), Tag.GetName(), EventScope, RenderCommandsChannel, !GCycleStatsShouldEmitNamedEvents);
		FScopeCycleCounter Scope(Tag.GetStatId(), true);
		ExecuteCommand_NoMarker(RHICmdList, Function);
	}
	else
	{
		ExecuteCommand_NoMarker(RHICmdList, Function);
	}
}

//////////////////////////////////////////////////////////////////////////

FRenderThreadCommandPipe FRenderThreadCommandPipe::Instance;

void FRenderThreadCommandPipe::EnqueueAndLaunch(FRenderCommandList* CommandList)
{
	EnqueueAndLaunch([this, CommandList](FRHICommandListImmediate&)
		{
			ExecuteCommands(CommandList);

		}, UE::RenderCommandPipe::GetExecuteCommandListTag());
}

void FRenderThreadCommandPipe::EnqueueAndLaunch(TUniqueFunction<void(FRHICommandListImmediate&)>&& Function, const FRenderCommandTag& Tag)
{
#if WITH_STATE_STREAM
	if (RenderCommandStateStream_AddCommand(Function, Tag))
		return;
#endif

	Mutex.Lock();
	const bool bWasEmpty = Context->CommandList.Enqueue(FRenderCommandFunctionVariant(TInPlaceType<TUniqueFunction<void(FRHICommandListImmediate&)>>(), MoveTemp(Function)), Tag);
	Mutex.Unlock();

	if (bWasEmpty)
	{
		TGraphTask<TFunctionGraphTaskImpl<void(), ESubsequentsMode::FireAndForget>>::CreateTask().ConstructAndDispatchWhenReady([this, ContextToConsume = Context]() mutable
			{
				Mutex.Lock();
				const bool bDeleteAfterExecute = ContextToConsume->bDeleteAfterExecute;
				FContext ContextToExecute(MoveTemp(*ContextToConsume));
				Mutex.Unlock();

				ContextToExecute.CommandList.Close();
				ExecuteCommands(ContextToExecute.CommandList);

				if (bDeleteAfterExecute)
				{
					delete ContextToConsume;
				}

			}, TStatId(), ENamedThreads::GetRenderThread());
	}
}

void FRenderThreadCommandPipe::ExecuteCommands(UE::RenderCommandPipe::FCommandList& CommandList)
{
	CommandList.ConsumeCommands([&RHICmdList = GetImmediateCommandList_ForRenderCommand()](FRenderCommandFunctionVariant&& Function, const FRenderCommandTag& Tag)
		{
			ExecuteCommand(&RHICmdList, Function, Tag);
		});
}

void FRenderThreadCommandPipe::ExecuteCommands(FRenderCommandList* CommandList)
{
	// Wait for recording of commands to be complete prior to replay.
	if (const UE::Tasks::FTask* DispatchTask = CommandList->TryGetDispatchTask())
	{
		DispatchTask->Wait();
	}

	// Execute commands for pipes that failed to enqueue due to not being in a recording state.
	if (!CommandList->PipeEnqueueFailedBits.IsEmpty())
	{
		for (TConstSetBitIterator<FRenderCommandPipeBitArrayAllocator> BitIt(CommandList->PipeEnqueueFailedBits); BitIt; ++BitIt)
		{
			ExecuteCommands(CommandList->Get(BitIt.GetIndex()));
		}
	}

	ExecuteCommands(CommandList->GetRenderThread());
	CommandList->ReleasePipeRef();
}

//////////////////////////////////////////////////////////////////////////

class FRenderCommandPipeRegistry
{
public:
	static TLinkedList<FRenderCommandPipe*>*& GetGlobalList()
	{
		static TLinkedList<FRenderCommandPipe*>* GlobalList = nullptr;
		return GlobalList;
	}

	void Initialize()
	{
		AllPipes.Reset();

		for (TLinkedList<FRenderCommandPipe*>::TIterator PipeIt(GetGlobalList()); PipeIt; PipeIt.Next())
		{
			FRenderCommandPipe* Pipe = *PipeIt;
			Pipe->SetEnabled(Pipe->ConsoleVariable->GetBool());
			Pipe->Index = AllPipes.Num();

			AllPipes.Emplace(*PipeIt);
		}
	}

	void StartRecording()
	{
		SerializeCommandList();

		if (GRenderCommandPipeMode != ERenderCommandPipeMode::All || !GIsThreadedRendering)
		{
			return;
		}

		FRenderCommandPipeBitArray PipeBits;
		PipeBits.Init(true, AllPipes.Num());
		StartRecording(PipeBits);
	}

	void StartRecording(const FRenderCommandPipeBitArray& PipeBits)
	{
		SerializeCommandList();

		if (GRenderCommandPipeMode != ERenderCommandPipeMode::All || !GIsThreadedRendering || PipeBits.IsEmpty())
		{
			return;
		}

		SCOPED_NAMED_EVENT(FRenderCommandPipe_StartRecording, FColor::Magenta);

		check(PipeBits.Num() == AllPipes.Num());

		UE::TScopeLock Lock(Mutex);

		bool bAnyPipesToStartRecording = false;

		for (FRenderCommandPipeSetBitIterator BitIt(PipeBits); BitIt; ++BitIt)
		{
			FRenderCommandPipe* Pipe = AllPipes[BitIt.GetIndex()];

			if (Pipe->bEnabled && !Pipe->bRecording)
			{
				bAnyPipesToStartRecording = true;
				break;
			}
		}

		if (!bAnyPipesToStartRecording)
		{
			return;
		}

		UE_RENDER_COMMAND_BEGIN_REGION(UE_RENDER_COMMAND_PIPE_RECORD_REGION);

		UE::Tasks::FTaskEvent TaskEvent{ UE_SOURCE_LOCATION };
		int32 NumPipesToStartRecording = 0;

		FRenderCommandPipeBitArray PipesToStartRecordingBits;
		PipesToStartRecordingBits.Init(false, PipeBits.Num());

		for (FRenderCommandPipeSetBitIterator BitIt(PipeBits); BitIt; ++BitIt)
		{
			FRenderCommandPipe* Pipe = AllPipes[BitIt.GetIndex()];

			if (Pipe->bEnabled && !Pipe->bRecording)
			{
				Pipe->bRecording = true;
				NumPipesToStartRecording++;
				PipesToStartRecordingBits[BitIt.GetIndex()] = true;

				UE::TScopeLock PipeLock(Pipe->Mutex);
				Pipe->RecordTask = TaskEvent;
			}
		}

		NumPipesRecording += NumPipesToStartRecording;

		ENQUEUE_RENDER_COMMAND(RenderCommandPipe_Start)([this, TaskEvent = MoveTemp(TaskEvent), PipesToStartRecordingBits, NumPipesToStartRecording](FRHICommandListImmediate&) mutable
			{
				RHIResourceLifetimeAddRef(NumPipesToStartRecording);
				NumPipesReplaying += NumPipesToStartRecording;
				TaskEvent.Trigger();

				for (FRenderCommandPipeSetBitIterator BitIt(PipesToStartRecordingBits); BitIt; ++BitIt)
				{
					AllPipes[BitIt.GetIndex()]->bReplaying = true;
				}
			});
	}

	FRenderCommandPipeBitArray StopRecording()
	{
		SerializeCommandList();

		UE::TScopeLock Lock(Mutex);
		if (!NumPipesRecording)
		{
			return {};
		}

		FRenderCommandPipeBitArray PipeBits;
		PipeBits.Init(false, AllPipes.Num());

		for (int32 PipeIndex = 0; PipeIndex < AllPipes.Num(); ++PipeIndex)
		{
			if (FRenderCommandPipe* Pipe = AllPipes[PipeIndex]; Pipe->bRecording)
			{
				PipeBits[PipeIndex] = true;
			}
		}

		StopRecording(PipeBits);
		return PipeBits;
	}

	FRenderCommandPipeBitArray StopRecording(TConstArrayView<FRenderCommandPipe*> Pipes)
	{
		SerializeCommandList();

		if (Pipes.IsEmpty())
		{
			return {};
		}

		UE::TScopeLock Lock(Mutex);
		if (!NumPipesRecording)
		{
			return {};
		}

		bool bAnyPipesToStopRecording = false;
		FRenderCommandPipeBitArray PipeBits;
		PipeBits.Init(false, AllPipes.Num());

		for (FRenderCommandPipe* Pipe : Pipes)
		{
			if (Pipe->bRecording)
			{
				PipeBits[Pipe->Index] = true;
				bAnyPipesToStopRecording = true;
			}
		}

		if (!bAnyPipesToStopRecording)
		{
			return {};
		}

		StopRecording(PipeBits);
		return PipeBits;
	}

	TConstArrayView<FRenderCommandPipe*> GetPipes() const
	{
		return AllPipes;
	}

	bool IsRecording() const
	{
		ensureMsgf(!FTaskTagScope::IsCurrentTag(ETaskTag::EParallelRenderingThread) && !FTaskTagScope::IsCurrentTag(ETaskTag::ERenderingThread),
			TEXT("IsRecording() is not valid from the render thread timeline."));

		return NumPipesRecording > 0;
	}

	bool IsReplaying() const
	{
		ensure(IsInParallelRenderingThread());
		return NumPipesReplaying > 0;
	}

private:
	void SerializeCommandList()
	{
		// Serialize all pipes in the active command list as we can't split the work.
		if (FRenderCommandList* CommandList = FRenderCommandList::GetInstanceTLS())
		{
			CommandList->bSerialized = 1;
		}
	}

	void StopRecording(const FRenderCommandPipeBitArray& PipeBits)
	{
		SCOPED_NAMED_EVENT(FRenderCommandPipe_StopRecording, FColor::Magenta);

		UE::Tasks::FTaskEvent TaskEvent{ UE_SOURCE_LOCATION };
		uint32 NumPipesToStopRecording = 0;

		for (FRenderCommandPipeSetBitIterator BitIt(PipeBits); BitIt; ++BitIt)
		{
			FRenderCommandPipe* Pipe = AllPipes[BitIt.GetIndex()];
			check(Pipe->bRecording);
			Pipe->bRecording = false;
			NumPipesToStopRecording++;

			Pipe->Mutex.Lock();

			Pipe->ResetContext();
			TaskEvent.AddPrerequisites(Pipe->RecordTask);
			Pipe->RecordTask = {};
		}

		NumPipesRecording -= NumPipesToStopRecording;
		TaskEvent.Trigger();

		ENQUEUE_RENDER_COMMAND(RenderCommandPipe_Stop)([this, PipeBits, TaskEvent = MoveTemp(TaskEvent), NumPipesToStopRecording](FRHICommandListImmediate& RHICmdList)
			{
				TArray<FRHICommandListImmediate::FQueuedCommandList, FConcurrentLinearArrayAllocator> QueuedCommandLists;
				QueuedCommandLists.Reserve(NumPipesToStopRecording);
				TaskEvent.Wait();

				for (FRenderCommandPipeSetBitIterator BitIt(PipeBits); BitIt; ++BitIt)
				{
					FRenderCommandPipe* Pipe = AllPipes[BitIt.GetIndex()];

					if (Pipe->RHICmdList)
					{
						Pipe->RHICmdList->FinishRecording();
						QueuedCommandLists.Emplace(Pipe->RHICmdList);
						Pipe->RHICmdList = nullptr;
					}

					Pipe->bReplaying = false;
				}

				NumPipesReplaying -= NumPipesToStopRecording;

				RHICmdList.QueueAsyncCommandListSubmit(QueuedCommandLists);
				RHIResourceLifetimeReleaseRef(RHICmdList, NumPipesToStopRecording);
			});

		// Wait to unlock the mutex until the sync command has been submitted to the render thread. This avoids
		// race conditions where a command meant for a specific pipe might be inserted to the render thread pipe
		// prior to the actual wait command.
		for (FRenderCommandPipeSetBitIterator BitIt(PipeBits); BitIt; ++BitIt)
		{
			AllPipes[BitIt.GetIndex()]->Mutex.Unlock();
		}

		UE_RENDER_COMMAND_END_REGION(UE_RENDER_COMMAND_PIPE_RECORD_REGION);
	}

	UE::FMutex Mutex;
	TArray<FRenderCommandPipe*> AllPipes;
	uint32 NumPipesRecording = 0;
	uint32 NumPipesReplaying = 0;
};

static FRenderCommandPipeRegistry GRenderCommandPipeRegistry;

inline bool HasBitsSet(const FRenderCommandPipeBitArray& Bits)
{
	for (FRenderCommandPipeBitArray::FConstWordIterator It(Bits); It; ++It)
	{
		if (It.GetWord() != 0)
		{
			return true;
		}
	}
	return false;
}

namespace UE::RenderCommandPipe
{
	static thread_local FRenderCommandPipe* ReplayingPipe = nullptr;
	static FStopRecordingDelegate StopRecordingDelegate;

	void Initialize()
	{
		GRenderCommandPipeRegistry.Initialize();
	}

	bool IsRecording()
	{
		return GRenderCommandPipeRegistry.IsRecording();
	}

	bool IsReplaying()
	{
		return GRenderCommandPipeRegistry.IsReplaying();
	}

	bool IsReplaying(const FRenderCommandPipe& Pipe)
	{
		return ReplayingPipe == &Pipe;
	}

	void StartRecording()
	{
		GRenderCommandPipeRegistry.StartRecording();
	}

	void StartRecording(const FRenderCommandPipeBitArray& PipeBits)
	{
		GRenderCommandPipeRegistry.StartRecording(PipeBits);
	}

	FRenderCommandPipeBitArray StopRecording()
	{
		FRenderCommandPipeBitArray PipeBits = GRenderCommandPipeRegistry.StopRecording();
		GetStopRecordingDelegate().Broadcast(PipeBits);
		return PipeBits;
	}

	FRenderCommandPipeBitArray StopRecording(TConstArrayView<FRenderCommandPipe*> Pipes)
	{
		FRenderCommandPipeBitArray PipeBits = GRenderCommandPipeRegistry.StopRecording(Pipes);
		GetStopRecordingDelegate().Broadcast(PipeBits);
		return PipeBits;
	}

	TConstArrayView<FRenderCommandPipe*> GetPipes()
	{
		return GRenderCommandPipeRegistry.GetPipes();
	}

	FStopRecordingDelegate& GetStopRecordingDelegate()
	{
		return StopRecordingDelegate;
	}

	FSyncScope::FSyncScope()
	{
		PipeBits = StopRecording();

#if UE_TRACE_ENABLED
		if (HasBitsSet(PipeBits))
		{
			UE_RENDER_COMMAND_BEGIN_REGION(UE_RENDER_COMMAND_PIPE_SYNC_REGION);
		}
#endif
	}

	FSyncScope::FSyncScope(TConstArrayView<FRenderCommandPipe*> Pipes)
	{
		PipeBits = StopRecording(Pipes);

#if UE_TRACE_ENABLED
		if (HasBitsSet(PipeBits))
		{
			UE_RENDER_COMMAND_BEGIN_REGION(UE_RENDER_COMMAND_PIPE_SYNC_REGION);
		}
#endif
	}

	FSyncScope::~FSyncScope()
	{
#if UE_TRACE_ENABLED
		if (HasBitsSet(PipeBits))
		{
			UE_RENDER_COMMAND_END_REGION(UE_RENDER_COMMAND_PIPE_SYNC_REGION);
		}
#endif

		StartRecording(PipeBits);
	}
}

//////////////////////////////////////////////////////////////////////////

FRenderCommandPipe::FRenderCommandPipe(const TCHAR* InName, ERenderCommandPipeFlags Flags, const TCHAR* CVarName, const TCHAR* CVarDescription)
	: Name(InName)
	, GlobalListLink(this)
	, ConsoleVariable(CVarName, !EnumHasAnyFlags(Flags, ERenderCommandPipeFlags::Disabled), CVarDescription, FConsoleVariableDelegate::CreateLambda([this](IConsoleVariable* Variable)
		{
			SetEnabled(Variable->GetBool());
		}))
{
#if !UE_SERVER
	GlobalListLink.LinkHead(FRenderCommandPipeRegistry::GetGlobalList());
#endif
}

void FRenderCommandPipe::ExecuteCommand(FRenderCommandFunctionVariant&& FunctionVariant, const FRenderCommandTag& Tag)
{
	if (!RHICmdList && FunctionVariant.IsType<TUniqueFunction<void(FRHICommandList&)>>())
	{
		RHICmdList = new FRHICommandList();
		RHICmdList->SwitchPipeline(ERHIPipeline::Graphics);
	}

	::ExecuteCommand(RHICmdList, FunctionVariant, Tag);
}

void FRenderCommandPipe::EnqueueAndLaunch(FRenderCommandList* CommandList)
{
	NumInFlightCommandLists.fetch_add(1, std::memory_order_relaxed);

	auto ExecuteLambda = [this, CommandList]
		{
			TRACE_CPUPROFILER_EVENT_SCOPE_ON_CHANNEL_STR("RenderCommandPipe ReplayCommands", RenderCommandsChannel);
			SCOPED_NAMED_EVENT_TCHAR(Name, FColor::Magenta);
			FTaskTagScope Scope(ETaskTag::EParallelRenderingThread);

			ExecuteCommands(CommandList->Get(this));
			NumInFlightCommandLists.fetch_sub(1, std::memory_order_relaxed);
			CommandList->ReleasePipeRef();
		};

	if (const UE::Tasks::FTask* DispatchTask = CommandList->TryGetDispatchTask())
	{
		ResetContext();

		RecordTask = UE::Tasks::Launch(Name, MoveTemp(ExecuteLambda), MakeArrayView<UE::Tasks::FTask>({ RecordTask, *DispatchTask }));
	}
	else
	{
		EnqueueAndLaunch(MoveTemp(ExecuteLambda), UE::RenderCommandPipe::GetExecuteCommandListTag());
	}
}

void FRenderCommandPipe::EnqueueAndLaunch(FRenderCommandFunctionVariant&& FunctionVariant, const FRenderCommandTag& Tag)
{
	ensureMsgf(!UE::RenderCommandPipe::ReplayingPipe, TEXT("Attempting to launch render command to render command pipe %s from another pipe %s"), Name, UE::RenderCommandPipe::ReplayingPipe->Name);

	bool bWasEmpty = Context->CommandList.Enqueue(MoveTemp(FunctionVariant), Tag);
	NumInFlightCommands.fetch_add(1, std::memory_order_relaxed);

	if (bWasEmpty)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_ON_CHANNEL_STR("RenderCommandPipe LaunchTask", RenderCommandsChannel);

		RecordTask = UE::Tasks::Launch(Name, [this, ContextToConsume = Context]
			{
				TRACE_CPUPROFILER_EVENT_SCOPE_ON_CHANNEL_STR("RenderCommandPipe ReplayCommands", RenderCommandsChannel)
					SCOPED_NAMED_EVENT_TCHAR(Name, FColor::Magenta);
				FTaskTagScope Scope(ETaskTag::EParallelRenderingThread);

				Mutex.Lock();
				const bool bDeleteAfterExecute = ContextToConsume->bDeleteAfterExecute;
				FContext ContextToExecute(MoveTemp(*ContextToConsume));
				Mutex.Unlock();

				ContextToExecute.CommandList.Close();
				ExecuteCommands(ContextToExecute.CommandList);

				const int32 NumCommands = ContextToExecute.CommandList.NumCommands();
				const int32 LocalNumInFlightCommands = NumInFlightCommands.fetch_sub(NumCommands, std::memory_order_release) - NumCommands;
				check(LocalNumInFlightCommands >= 0);

				if (bDeleteAfterExecute)
				{
					delete ContextToConsume;
				}

			}, RecordTask);
	}
}

void FRenderCommandPipe::ExecuteCommands(UE::RenderCommandPipe::FCommandList& CommandList)
{
	FRenderCommandPipe* const PreviousReplayingPipe = UE::RenderCommandPipe::ReplayingPipe;
	UE::RenderCommandPipe::ReplayingPipe = this;

	int32 NumCommands = CommandList.NumCommands();

	CommandList.ConsumeCommands([this](FRenderCommandFunctionVariant&& FunctionVariant, const FRenderCommandTag& Tag)
		{
			ExecuteCommand(MoveTemp(FunctionVariant), Tag);
		});

	UE::RenderCommandPipe::ReplayingPipe = PreviousReplayingPipe;
}
