// Copyright Epic Games, Inc. All Rights Reserved.

#include "RenderingThread.h"
#include "Async/ParallelFor.h"

thread_local FRenderCommandList* FRenderCommandList::InstanceTLS = nullptr;

FRenderCommandList* FRenderCommandList::GetInstanceTLS()
{
	return InstanceTLS;
}

FRenderCommandList* FRenderCommandList::SetInstanceTLS(FRenderCommandList* CommandList)
{
	FRenderCommandList* PreviousCommandList = InstanceTLS;
	InstanceTLS = CommandList;
	return PreviousCommandList;
}

//////////////////////////////////////////////////////////////////////////

FRenderCommandList::FRecordScope::FRecordScope(FRenderCommandList* InCommandList, EStopRecordingAction InStopAction)
	: CommandList(InCommandList)
	, StopAction(InStopAction)
{
#if DO_CHECK
	if (CommandList)
	{
		CommandList->NumRecordScopeRefs++;
		checkf(CommandList->NumRecordScopeRefs != 0, TEXT("FRecordScope is either being nested recursively on the same command list or is recording from multiple threads."));
		check(CommandList->bRecording);
	}
#endif

	PreviousCommandList = SetInstanceTLS(InCommandList);
}

FRenderCommandList::FRecordScope::~FRecordScope()
{
	SetInstanceTLS(PreviousCommandList);

	if (CommandList)
	{
#if DO_CHECK
		CommandList->NumRecordScopeRefs--;
#endif

		if (StopAction == EStopRecordingAction::Close)
		{
			CommandList->Close();
		}
		else if (StopAction == EStopRecordingAction::Submit)
		{
			CommandList->Close();
			FRenderCommandDispatcher::Submit(CommandList);
		}
	}
}

FRenderCommandList::FFlushScope::FFlushScope()
{
	CommandList = SetInstanceTLS(nullptr);

	if (CommandList)
	{
		CommandList->Flush();
	}
}

FRenderCommandList::FFlushScope::~FFlushScope()
{
	if (CommandList)
	{
		SetInstanceTLS(CommandList);
	}
}

//////////////////////////////////////////////////////////////////////////

FRenderCommandList::FParallelForContext::FParallelForContext(FRenderCommandList* InRootCommandList, int32 NumTasks, int32 BatchSize, EParallelForFlags Flags)
	: FParallelForContext(InRootCommandList, ParallelForImpl::GetNumberOfThreadTasks(NumTasks, BatchSize, Flags))
{
}

FRenderCommandList::FParallelForContext::FParallelForContext(FRenderCommandList* InRootCommandList, int32 NumContexts)
	: RootCommandList(InRootCommandList)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(FRenderCommandList::FParallelForContext::Init);

	if (!RootCommandList)
	{
		RootCommandList = FRenderCommandList::Create(ERenderCommandListFlags::CloseOnSubmit);
		bSubmitRootCommandList = true;
	}

	TaskCommandLists.Reserve(NumContexts);

	for (int32 Index = 0; Index < NumContexts - 1; ++Index)
	{
		TaskCommandLists.Emplace(FRenderCommandList::Create(ERenderCommandListFlags::CloseOnSubmit));
	}

	TaskCommandLists.Emplace(RootCommandList);
}

void FRenderCommandList::FParallelForContext::Submit()
{
	if (RootCommandList)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE(FRenderCommandList::FParallelForContext::Submit);

		for (int32 Index = 0; Index < TaskCommandLists.Num() - 1; ++Index)
		{
			FRenderCommandDispatcher::Submit(TaskCommandLists[Index], RootCommandList);
		}

		if (bSubmitRootCommandList)
		{
			FRenderCommandDispatcher::Submit(RootCommandList);
		}

		RootCommandList = nullptr;
		TaskCommandLists.Empty();
	}
}

//////////////////////////////////////////////////////////////////////////

FRenderCommandList::FRenderCommandList(ERenderCommandListFlags InFlags, EPageSize PageSize)
	: Allocator(PageSize)
	, Flags(InFlags)
{
	if (!EnumHasAnyFlags(Flags, ERenderCommandListFlags::CloseOnSubmit))
	{
		DispatchTaskEvent.Emplace(UE_SOURCE_LOCATION);
		Init();
	}
}

FRenderCommandList::~FRenderCommandList()
{
	FRenderCommandList* Child = Children.Head;
	while (Child)
	{
		FRenderCommandList* NextChild = Child->NextSibling;
		delete Child;
		Child = NextChild;
	}
	Children = {};

	if (DispatchTaskEvent)
	{
		DispatchTaskEvent->Trigger();
	}
}

void FRenderCommandList::Flush()
{
	if (bSubmitted || !bInitialized || !EnumHasAnyFlags(Flags, ERenderCommandListFlags::CloseOnSubmit))
	{
		return;
	}

	FRenderCommandList* FlushCommandList = FRenderCommandList::Create(ERenderCommandListFlags::CloseOnSubmit);
	FlushCommandList->Allocator = MoveTemp(Allocator);
	FlushCommandList->CommandLists = MoveTemp(CommandLists);
	FlushCommandList->DispatchTaskEvent = MoveTemp(DispatchTaskEvent);
	FlushCommandList->bInitialized = true;
	FlushCommandList->Children = Children;

	FRenderCommandDispatcher::Submit(FlushCommandList);

	bInitialized = false;
	Children = {};
}

void FRenderCommandList::Init()
{
	check(!bInitialized);
	bInitialized = true;
	const int32 NumCommandLists = UE::RenderCommandPipe::GetPipes().Num() + 1;
	CommandLists.Reserve(NumCommandLists);
	for (int32 Index = 0; Index < NumCommandLists; ++Index)
	{
		CommandLists.Emplace(Allocator);
	}
}

void FRenderCommandList::Close()
{
#if DO_CHECK
	checkf(NumRecordScopeRefs == 0, TEXT("Close called on command list while FRecordScope is active!"));
	checkf(bRecording, TEXT("Close has already been called on this command list."));
#endif

	bRecording = false;

	for (UE::RenderCommandPipe::FCommandList& CommandList : CommandLists)
	{
		CommandList.Close();
	}

	if (DispatchTaskEvent)
	{
		DispatchTaskEvent->Trigger();
	}
}

void FRenderCommandList::Submit(FRenderCommandList* InParent)
{
	const bool bCloseOnSubmit = EnumHasAnyFlags(Flags, ERenderCommandListFlags::CloseOnSubmit);

	if (bCloseOnSubmit)
	{
		if (bRecording)
		{
			Close();
		}

		if (!bInitialized)
		{
			delete this;
			return;
		}
	}

	checkf(!bSubmitted, TEXT("FRenderCommandList::Submit cannot be called multiple times."));
	bSubmitted = true;

	Parent = InParent;

	if (!Parent)
	{
		Parent = GetInstanceTLS();
	}

	TConstArrayView<FRenderCommandPipe*> Pipes = UE::RenderCommandPipe::GetPipes();

	// Submit into parent.
	if (Parent)
	{
		bool bSkippedAllLists = true;

		for (int32 Index = 0; Index < Pipes.Num(); ++Index)
		{
			FRenderCommandPipe* Pipe = Pipes[Index];
			UE::RenderCommandPipe::FCommandList& CommandList = CommandLists[Index];
			const bool bSkipEmptyList = bCloseOnSubmit && CommandList.IsEmpty();

			if (!bSkipEmptyList)
			{
				Parent->Get(Pipe).Enqueue(&CommandList);
				bSkippedAllLists = false;
			}
		}

		UE::RenderCommandPipe::FCommandList& CommandList = CommandLists.Last();
		const bool bSkipEmptyList = bCloseOnSubmit && CommandList.IsEmpty();

		if (!bSkipEmptyList)
		{
			Parent->GetRenderThread().Enqueue(&CommandList);
			bSkippedAllLists = false;
		}

		if (bSkippedAllLists)
		{
			delete this;
			return;
		}

		check(Parent->bInitialized);

		// Parent takes ownership of the child command list for deletion purposes.
		if (Parent->Children.Tail)
		{
			Parent->Children.Tail->NextSibling = this;
			Parent->Children.Tail = this;
		}
		else
		{
			Parent->Children.Head = Parent->Children.Tail = this;
		}

		// If have a valid recording task event, either we are still recording or one of our children is, so we have to propagate the event up to the parent.
		if (DispatchTaskEvent)
		{
			if (!Parent->DispatchTaskEvent)
			{
				check(!Parent->bSubmitted && Parent->bRecording);
				Parent->DispatchTaskEvent.Emplace(UE_SOURCE_LOCATION);
			}

			Parent->DispatchTaskEvent->AddPrerequisites(*DispatchTaskEvent);
		}
	}
	// Submit into command pipe.
	else
	{
		// Start by setting the maximum amount of refs. This has to happen first to avoid race with release on pipe threads.
		const int32 MaxNumRefs = CommandLists.Num();
		NumPipeRefs.fetch_add(MaxNumRefs, std::memory_order_relaxed);

		int32 NumEnqueues = 0;
		int32 NumPipeEnqueueFailed = 0;

		for (int32 Index = 0; Index < Pipes.Num(); ++Index)
		{
			FRenderCommandPipe* Pipe = Pipes[Index];
			UE::RenderCommandPipe::FCommandList& CommandList = CommandLists[Index];
			const bool bSkipEmptyList = bCloseOnSubmit && CommandList.IsEmpty();

			if (!bSkipEmptyList)
			{
					if (!bSerialized && Pipes[Index]->Enqueue(this))
				{
					NumEnqueues++;
				}
				else
				{
					if (PipeEnqueueFailedBits.IsEmpty())
					{
						PipeEnqueueFailedBits.Init(false, Pipes.Num());
					}
					PipeEnqueueFailedBits[Index] = true;
					NumPipeEnqueueFailed++;
				}
			}
		}

		UE::RenderCommandPipe::FCommandList& CommandList = CommandLists.Last();
		const bool bSkipEmptyList = bCloseOnSubmit && CommandList.IsEmpty();

		if (NumPipeEnqueueFailed > 0 || !bSkipEmptyList)
		{
			FRenderThreadCommandPipe::Enqueue(this);
			NumEnqueues++;
		}

		const int32 NumRefsSkipped = MaxNumRefs - NumEnqueues;
		ReleasePipeRefs(NumRefsSkipped);
	}
}

