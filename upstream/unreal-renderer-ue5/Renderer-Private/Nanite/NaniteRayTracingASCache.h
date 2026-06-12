// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Containers/RingBuffer.h"
#include "SpanAllocator.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphResources.h"
#include "RenderGraphUtils.h"
#include "RHIGPUReadback.h"
#include "NaniteRayTracingDefinitions.h"
#include "NaniteDefinitions.h"

#if RHI_RAYTRACING

namespace Nanite
{

class FNaniteRayTracingBLASCache
{
public:
	FNaniteRayTracingBLASCache(uint32 InMaxSizeBytes);
	~FNaniteRayTracingBLASCache();

	// Process any ready readback buffers, updating the cache's CPU-side allocation state.
	// Call before UploadAllocationTable to ensure consistent state.
	void ProcessReadbacks();

	// Uploads CachedASes to a transient GPU structured buffer and returns an SRV.
	FRDGBufferSRV* UploadAllocationTable(FRDGBuilder& GraphBuilder) const;

	FRDGBufferRef RegisterOrCreateMetadataBuffer(FRDGBuilder& GraphBuilder);

	// Creates and clears the per-frame cache request buffer for GPU to write into
	FRDGBufferRef CreateRequestBuffer(FRDGBuilder& GraphBuilder, uint32 NumRootPages);

	// Submits the request buffer for readback after GPU has written to it
	void SubmitRequestBuffer(FRDGBuilder& GraphBuilder, FRDGBufferRef RequestBuffer, uint32 NumRootPages);

	uint32 GetMaxSize() const { return MaxSizeBytes; }

	TRefCountPtr<FRDGPooledBuffer>& GetBuffer() { return Buffer; }

	void InvalidateEntry(uint32 RuntimeResourceId) { InvalidateEntry(GetEntry(RuntimeResourceId)); }

private:
	// Process GPU readback for one frame.
	// Evicts entries that are no longer requested or whose key/size no longer matches.
	// Allocates space for any new requests.
	void ProcessCacheRequests(TConstArrayView<FNaniteRayTracingASCacheRequest> Requests);

	FNaniteRayTracingASCacheEntry& GetEntry(uint32 RuntimeResourceId);

	static void InvalidateEntry(FNaniteRayTracingASCacheEntry& Entry);

	// Indexed by root page index; sized to NANITE_MAX_GPU_PAGES. Empty entries have ByteSize == 0.
	TArray<FNaniteRayTracingASCacheEntry> 	CachedASes;

	FSpanAllocator                          Allocator;
	uint32 									MaxSizeBytes = 0;

	TRefCountPtr<FRDGPooledBuffer>          Buffer;
	TRefCountPtr<FRDGPooledBuffer>          BLASCacheMetadataBuffer;

	struct FReadbackInfo
	{
		TUniquePtr<FRHIGPUBufferReadback> Buffer;
		uint32 Size = 0;
	};

	TRingBuffer<FReadbackInfo> PendingReadbacks;
	TArray<FReadbackInfo> AvailableReadbacks;
};

} // namespace Nanite

#endif // RHI_RAYTRACING
