// Copyright Epic Games, Inc. All Rights Reserved.

#include "NaniteRayTracingASCache.h"

#if RHI_RAYTRACING

#include "RendererModule.h"

namespace Nanite
{

static constexpr int32 MaxReadbackBuffers = 4;

FNaniteRayTracingBLASCache::FNaniteRayTracingBLASCache(uint32 InMaxSizeBytes)
	: MaxSizeBytes(AlignArbitrary(InMaxSizeBytes, GRHIGlobals.RayTracing.AccelerationStructureAlignment))
{
	CachedASes.SetNumZeroed(NANITE_MAX_GPU_PAGES);

	FRDGBufferDesc Desc;
	Desc.Usage = EBufferUsageFlags::AccelerationStructure;
	Desc.BytesPerElement = GRHIGlobals.RayTracing.AccelerationStructureAlignment;
	Desc.NumElements = FMath::DivideAndRoundUp<uint32>(MaxSizeBytes, Desc.BytesPerElement);
	AllocatePooledBuffer(Desc, Buffer, TEXT("NaniteRayTracing.BLASCacheBuffer"));

	AvailableReadbacks.Reserve(MaxReadbackBuffers);
	PendingReadbacks.Reserve(MaxReadbackBuffers);
	for (int32 Index = 0; Index < MaxReadbackBuffers; ++Index)
	{
		FReadbackInfo Readback;
		Readback.Buffer = MakeUnique<FRHIGPUBufferReadback>(TEXT("NaniteRayTracing.CacheReadbackBuffer"));
		AvailableReadbacks.Add(MoveTemp(Readback));
	}
}

FNaniteRayTracingBLASCache::~FNaniteRayTracingBLASCache() = default;

FNaniteRayTracingASCacheEntry& FNaniteRayTracingBLASCache::GetEntry(uint32 RuntimeResourceId)
{
	return CachedASes[RuntimeResourceId & NANITE_MAX_GPU_PAGES_MASK];
}

void FNaniteRayTracingBLASCache::InvalidateEntry(FNaniteRayTracingASCacheEntry& Entry)
{
	// Incrementing ensures the new value won't match stale GPU metadata, preventing false
	// cache hits on the first frame after reallocation.
	++Entry.UpdateSequenceId;

	// Avoid any potential issues on wraparound matching the cleared entry value
	if (Entry.UpdateSequenceId == 0U)
	{
		++Entry.UpdateSequenceId;
	}
}

void FNaniteRayTracingBLASCache::ProcessReadbacks()
{
	while (!PendingReadbacks.IsEmpty() && PendingReadbacks.First().Buffer->IsReady())
	{
		FReadbackInfo Readback = PendingReadbacks.PopFrontValue();

		check((Readback.Size % sizeof(FNaniteRayTracingASCacheRequest)) == 0);
		uint32 Count = Readback.Size / sizeof(FNaniteRayTracingASCacheRequest);

		auto ReadbackBufferPtr = (const FNaniteRayTracingASCacheRequest*)Readback.Buffer->Lock(Readback.Size);
		ProcessCacheRequests(MakeArrayView(ReadbackBufferPtr, Count));
		Readback.Buffer->Unlock();

		AvailableReadbacks.Add(MoveTemp(Readback));
	}
}

FRDGBufferRef FNaniteRayTracingBLASCache::CreateRequestBuffer(FRDGBuilder& GraphBuilder, uint32 NumRootPages)
{
	const uint32 BufferElements = FMath::Min(FMath::RoundUpToPowerOfTwo(FMath::Max(NumRootPages, 1u)), (uint32)NANITE_MAX_GPU_PAGES);
	FRDGBufferDesc Desc = FRDGBufferDesc::CreateByteAddressDesc(sizeof(FNaniteRayTracingASCacheRequest) * BufferElements);
	Desc.Usage |= EBufferUsageFlags::SourceCopy;
	FRDGBufferRef RequestBuffer = GraphBuilder.CreateBuffer(Desc, TEXT("NaniteRayTracing.BLAS.CacheRequest"));
	AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(RequestBuffer), 0);
	return RequestBuffer;
}

void FNaniteRayTracingBLASCache::SubmitRequestBuffer(FRDGBuilder& GraphBuilder, FRDGBufferRef RequestBuffer, uint32 NumRootPages)
{
	check((RequestBuffer->Desc.Usage & EBufferUsageFlags::SourceCopy) == EBufferUsageFlags::SourceCopy);

	if (AvailableReadbacks.IsEmpty())
	{
		check(false);
		return;
	}

	FReadbackInfo Readback = AvailableReadbacks.Pop();
	FRHIGPUBufferReadback* ReadbackBufferRaw = Readback.Buffer.Get();
	Readback.Size = sizeof(FNaniteRayTracingASCacheRequest) * NumRootPages;

	AddReadbackBufferPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::CacheReadback"), RequestBuffer,
		[ReadbackBufferRaw, RequestBuffer](FRDGAsyncTask, FRHICommandList& RHICmdList)
		{
			ReadbackBufferRaw->EnqueueCopy(RHICmdList, RequestBuffer->GetRHI(), 0u);
		});

	PendingReadbacks.Add(MoveTemp(Readback));
}

void FNaniteRayTracingBLASCache::ProcessCacheRequests(TConstArrayView<FNaniteRayTracingASCacheRequest> Requests)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(FNaniteRayTracingBLASCache::ProcessCacheRequests);

	check(Requests.Num() <= NANITE_MAX_GPU_PAGES);

	// Do all evictions first to reduce fragmentation
	for (int32 RootPageIndex = 0; RootPageIndex < Requests.Num(); ++RootPageIndex)
	{
		const FNaniteRayTracingASCacheRequest Request = Requests[RootPageIndex];
		FNaniteRayTracingASCacheEntry& Entry = CachedASes[RootPageIndex];

		// NOTE: The CPU side is only concerned with allocation.
		if (Entry.ByteSize != 0)
		{
			if (Request.Size != Entry.ByteSize)
			{
				Allocator.Free(Entry.ByteOffset, Entry.ByteSize);
				// Preserve UpdateSequenceId across eviction so it stays ahead of the GPU metadata
				// and prevents false cache hits after reallocation (see allocation loop below).
				uint32 PrevSequenceId = Entry.UpdateSequenceId;
				Entry = {};
				Entry.UpdateSequenceId = PrevSequenceId;
			}
		}
	}

	// Now do any allocations
	for (int32 RootPageIndex = 0; RootPageIndex < Requests.Num(); ++RootPageIndex)
	{
		const FNaniteRayTracingASCacheRequest Request = Requests[RootPageIndex];
		FNaniteRayTracingASCacheEntry& Entry = CachedASes[RootPageIndex];

		// Allocate if there is an active request and the slot is empty
		if (Entry.ByteSize == 0 && Request.Size != 0)
		{
			check((Request.Size % GRHIGlobals.RayTracing.AccelerationStructureAlignment) == 0)
			const int32 AllocOffset = Allocator.Allocate(Request.Size);
			if (Allocator.GetMaxSize() > (int32)MaxSizeBytes)
			{
				//UE_LOGF(LogRenderer, Warning, "FNaniteRayTracingBLASCache [%d] failed to allocate %d, %d/%d", RootPageIndex, Request.Size, Allocator.GetSparselyAllocatedSize(), MaxSizeBytes);
				// If we went over the limit, undo that last one and bail
				Allocator.Free(AllocOffset, Request.Size);
			}
			else
			{
				Entry.ByteOffset = AllocOffset;
				Entry.ByteSize   = Request.Size;
				InvalidateEntry(Entry);
			}
		}
	}
}

FRDGBufferSRV* FNaniteRayTracingBLASCache::UploadAllocationTable(FRDGBuilder& GraphBuilder) const
{
	FRDGBufferRef TableBuffer = CreateStructuredBuffer(
		GraphBuilder, TEXT("NaniteRayTracing.BLASCache.AllocationTable"),
		MakeConstArrayView(CachedASes));
	return GraphBuilder.CreateSRV(TableBuffer);
}

FRDGBufferRef FNaniteRayTracingBLASCache::RegisterOrCreateMetadataBuffer(FRDGBuilder& GraphBuilder)
{
	if (!BLASCacheMetadataBuffer)
	{
		FRDGBufferDesc Desc = FRDGBufferDesc::CreateStructuredDesc(sizeof(FNaniteRayTracingASCacheMetadata), NANITE_MAX_GPU_PAGES);
		FRDGBufferRef MetadataBuffer = GraphBuilder.CreateBuffer(Desc, TEXT("NaniteRayTracing.BLASCache.Metadata"));
		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(MetadataBuffer), 0u);
		GraphBuilder.QueueBufferExtraction(MetadataBuffer, &BLASCacheMetadataBuffer);
		return MetadataBuffer;
	}
	return GraphBuilder.RegisterExternalBuffer(BLASCacheMetadataBuffer);
}

} // namespace Nanite

#endif // RHI_RAYTRACING
