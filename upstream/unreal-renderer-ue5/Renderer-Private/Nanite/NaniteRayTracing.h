// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "NaniteShared.h"
#include "NaniteRayTracingASCache.h"

#include "CoreMinimal.h"
#include "SpanAllocator.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphResources.h"
#include "RenderGraphUtils.h"
#include "RHIGPUReadback.h"
#include "Rendering/NaniteResources.h"

#include "MeshPassProcessor.h"
#include "InstanceCulling/InstanceCullingLoadBalancer.h"

#include "Experimental/Containers/SherwoodHashTable.h"

#if RHI_RAYTRACING

class FScene;
class FRayTracingScene;

namespace Nanite
{
	using FRayTracingLoadBalancer = TInstanceCullingLoadBalancer<SceneRenderingAllocator>;

	class FRayTracingManager : public FRenderResource
	{
	public:
		FRayTracingManager();
		~FRayTracingManager();

		void Initialize();
		void Shutdown();

		virtual void InitRHI(FRHICommandListBase& RHICmdList) override;
		virtual void ReleaseRHI() override;

		void Add(FPrimitiveSceneInfo* SceneInfo);
		void Remove(FPrimitiveSceneInfo* SceneInfo);
		void AddVisiblePrimitive(const FPrimitiveSceneInfo* SceneInfo);

		void RequestUpdates(const TMap<uint32, uint32>& InUpdateRequests, const TSet<FPageInfo>& InInstalledPages, const TSet<FPageInfo>& InUninstalledPages);

		void Update(FRHICommandListBase& RHICmdList);

		void UpdateStreaming(FRDGBuilder& GraphBuilder, TConstArrayView<FViewInfo> Views, FSceneUniformBuffer& SceneUniformBuffer, FIntPoint RasterTextureSize);

		// Dispatch compute shader to stream out mesh data for resources with update requests.
		void ProcessUpdateRequests(FRDGBuilder& GraphBuilder, FSceneUniformBuffer& SceneUniformBuffer);

		// Commit pending BLAS builds. This allocates a transient scratch buffer internally.
		bool ProcessBuildRequests(FRDGBuilder& GraphBuilder);

		void PostRender();

		void EndFrame();

		FRDGBufferSRV* GetAuxiliaryDataSRV(FRDGBuilder& GraphBuilder) const
		{
			return GraphBuilder.CreateSRV(GraphBuilder.RegisterExternalBuffer(AuxiliaryDataBuffer));
		}

		FRDGBufferSRV* GetBLASDataSRV(FRDGBuilder& GraphBuilder) const
		{
			return GetRayTracingMode() == ERayTracingMode::CLAS && BLASDataBuffer.IsValid() ? GraphBuilder.CreateSRV(GraphBuilder.RegisterExternalBuffer(BLASDataBuffer)) : nullptr;
		}

		// Returns acceleration structure buffers that the TLAS build implicitly reads via GPU virtual addresses.
		// These must be declared as dependencies in the TLAS build pass to ensure proper BVHWrite->BVHRead barriers.
		void GetTLASBuildDependencies(FRDGBuilder& GraphBuilder, FRDGBufferAccessArray& OutBufferAccesses) const;

		FRHIRayTracingGeometry* GetRayTracingGeometry(const FPrimitiveSceneInfo* SceneInfo) const;
		uint32 GetRayTracingGeometrySegmentCount(const FPrimitiveSceneInfo* SceneInfo) const;

		bool CheckModeChanged();

		TRDGUniformBufferRef<FNaniteRayTracingUniformParameters> GetUniformBuffer() const
		{
			return UniformBuffer;
		}

		FRHIUniformBuffer* GetUniformBufferRHI(FRDGBuilder& GraphBuilder)
		{
			return GraphBuilder.ConvertToExternalUniformBuffer(GetUniformBuffer());
		}

		void UpdateUniformBuffer(FRDGBuilder& GraphBuilder, bool bShouldRenderNanite);

	private:
		class FClusterStreamOutBuffers
		{
		public:
			FClusterStreamOutBuffers(uint32 MaxClusters);

			void Initialize(FRDGBuilder& GraphBuilder);
			void RequestScratch(FRDGBuilder& GraphBuilder, uint32 Size);
			void RequestStaging(FRDGBuilder& GraphBuilder, uint32 Size);
			void Cleanup();

			const uint32 MaxClusters;
			const uint32 MaxNumVertices;
			const uint32 MaxNumIndices;
			const uint32 MaxNumTriangles;

			FRDGBufferRef ClusterVertexBuffer;
			FRDGBufferRef ClusterIndexBuffer;
			FRDGBufferRef ClusterGeometryIndexAndFlagsBuffer;
			FRDGBufferRef ClusterCountBuffer;
			FRDGBufferRef ClusterBuildDescBuffer;

			FRayTracingClusterOperationInitializer ClusterBuildOpInitializer = {};
			FRayTracingClusterOperationSize ClusterBuildOpSize = {};
			FRDGBufferRef ScratchBuffer;
			FRDGBufferRef StagingBuffer;

		private:
			// Keep consistent sizes for these buffers frame to frame to avoid thrashing the buffer pool
			uint32 ScratchSize = 0;
			uint32 StagingSize = 0;
		};

		void ProcessPendingSegmentMappingUploads(FRDGBuilder& GraphBuilder);
		void ProcessUpdateRequestsModeStreamOut(FRDGBuilder& GraphBuilder, FSceneUniformBuffer& SceneUniformBuffer);
		void ProcessUpdateRequestsModeCLAS(FRDGBuilder& GraphBuilder, FSceneUniformBuffer& SceneUniformBuffer);

		void ResizeCLASBufferIfNeeded(FRDGBuilder& GraphBuilder);
		void ResizeBLASCacheIfNeeded(FRDGBuilder& GraphBuilder);

		void StreamOutClusters(
			FRDGBuilder& GraphBuilder,
			FGlobalShaderMap* ShaderMap,
			uint32 NumRequests,
			FRDGBufferRef RequestBuffer,
			FRDGBufferRef ResourceAddressesBuffer,
			FRDGBufferRef SegmentMappingBuffer);

		void CalculateReferenceErrors(
			FRDGBuilder& GraphBuilder,
			FSceneUniformBuffer& SceneUniformBuffer,
			FRDGBufferSRVRef PackedNaniteViews,
			const FRayTracingLoadBalancer::FGPUData& LoadBalancerGPUData);

		void BuildCLASForPages(FRDGBuilder& GraphBuilder,
							   const TArray<FPageInfo>& Pages,
							   FRDGBufferRef ResourceAddressesBufferRDG);

		void ProcessPendingInstalledPages(FRDGBuilder& GraphBuilder);
		void ProcessPendingUninstalledPages(FRDGBuilder& GraphBuilder);
		void ProcessPendingCLASBuildPages(FRDGBuilder& GraphBuilder);
		void ProcessPendingPages(FRDGBuilder& GraphBuilder);
		void UpdateBLAS(FRDGBuilder& GraphBuilder);
		void DebugModeCLAS(FRDGBuilder& GraphBuilder);
		void ResetPageInstallationState();

		uint32 GetRequestedCLASAllocationSize() const;
		uint32 GetMaxPageInstallBatchSize() const;

		struct FInternalData
		{
			TSet<FPrimitiveSceneInfo*> Primitives;
			uint32 RuntimeResourceID;
			uint32 HierarchyOffset;
			uint32 NumClusters;
			uint32 NumNodes;
			uint32 NumVertices;
			uint32 NumTriangles;
			uint32 NumMaterials;
			uint32 NumSegments;

			bool bAssembly;

			uint32 NumResidentClusters;
			uint32 NumResidentClustersUpdate;

			uint32 PrimitiveId;

			TArray<uint32> SegmentMapping;
			uint32 SegmentMappingOffset = INDEX_NONE;

			FDebugName DebugName;

			FRayTracingGeometryRHIRef RayTracingGeometryRHI;

			uint32 AuxiliaryDataOffset = INDEX_NONE;
			uint32 AuxiliaryDataSize = 0;

			uint32 StagingAuxiliaryDataOffset = INDEX_NONE;
			int32 BaseMeshDataOffset = -1;
			bool bUpdating = false;
		};

		struct FPendingBuild
		{
			FRayTracingGeometryRHIRef RayTracingGeometryRHI;
			uint32 GeometryId;
		};

		TMap<uint32, uint32> ResourceToRayTracingIdMap;
		TSparseArray<FInternalData*> Geometries;

		TSet<uint32> UpdateRequests;
		TSet<uint32> VisibleGeometries;
		TSet<const FPrimitiveSceneInfo*> VisiblePrimitives;

		TSet<uint32> PendingRemoves;

		TSet<FPageInfo> PendingInstalledPages;
		TSet<FPageInfo> PendingUninstalledPages;
		TSet<FPageInfo> PendingCLASBuildPages;

		TRefCountPtr<FRDGPooledBuffer> AuxiliaryDataBuffer;
		FSpanAllocator AuxiliaryDataAllocator;

		TRefCountPtr<FRDGPooledBuffer> StagingAuxiliaryDataBuffer;

		TRefCountPtr<FRDGPooledBuffer> VertexBuffer;
		TRefCountPtr<FRDGPooledBuffer> IndexBuffer;

		TRefCountPtr<FRDGPooledBuffer> SegmentMappingBuffer;
		FSpanAllocator SegmentMappingAllocator;
		FRDGScatterUploadBuffer SegmentMappingUploadBuffer;
		TSet<uint32> PendingSegmentMappingUpload;
		uint32 PendingSegmentMappingUploadCount = 0;

		TRefCountPtr<FRDGPooledBuffer> ReferenceErrorsBuffer;

		TRefCountPtr<FRDGPooledBuffer> CLASResourceAddressesBuffer;
		TRefCountPtr<FRDGPooledBuffer> BLASResourceAddressesBuffer;

		TRefCountPtr<FRDGPooledBuffer> CLASBuffer;
		TRefCountPtr<FRDGPooledBuffer> CLASAllocatorBuffer;

		TRefCountPtr<FRDGPooledBuffer> PageDataBuffer;
		FRDGScatterUploadBuffer PageDataUploadBuffer;

		TRefCountPtr<FRDGPooledBuffer> CLASDataBuffer;
		FSpanAllocator CLASDataAllocator;
		FRDGScatterUploadBuffer CLASDataUploadBuffer;

		TSharedPtr<FClusterStreamOutBuffers> StreamOutBuffers;

		TSharedPtr<FNaniteRayTracingBLASCache> BLASCache;

		uint32 CLASMaxSize = 0;		// Max driver-reported CLAS size
		uint32 CLASPoolAllocationSize = 0;
		uint32 CLASPoolSize = 0;
		uint32 CLASBufferAllocatedSize = 0;
		bool bCLASBufferFull = false;

		TRefCountPtr<FRDGPooledBuffer> BLASStagingBuffer;

		TRefCountPtr<FRDGPooledBuffer> BLASDataBuffer;

		uint32 MaxGPUPageIndex = 0;
		uint32 MaxRootPageIndex = 0;

		struct FInstalledPage
		{
			uint32 BaseIndex;
			uint32 NumClusters;
			uint32 RuntimeResourceID;
		};

		TMap<uint32, FInstalledPage> InstalledPages;

		struct FCLASReadbackData
		{
			FRHIGPUBufferReadback* ReadbackBuffer = nullptr;
			uint32 AllocationSize = 0;
			uint32 PoolSize = 0;
		};

		TArray<FCLASReadbackData> CLASReadbackDatas;
		uint32 CLASReadbackBuffersWriteIndex = 0;
		uint32 CLASReadbackBuffersNumPending = 0;

		struct FReadbackData
		{
			FRHIGPUBufferReadback* MeshDataReadbackBuffer = nullptr;
			uint32 NumMeshDataEntries = 0;

			TArray<uint32> EntryGeometryId;
		};

		TArray<FReadbackData> ReadbackBuffers;
		uint32 ReadbackBuffersWriteIndex;
		uint32 ReadbackBuffersNumPending;

		// Geometries to be built this frame
		TArray<uint32> ScheduledBuilds;
		int32 ScheduledBuildsNumPrimitives = 0;

		// Geometries pending BLAS build due to r.RayTracing.Nanite.MaxBlasBuildsPerFrame throttling
		TArray<FPendingBuild> PendingBuilds;

		FRDGBufferRef StatsBufferRDG = nullptr;
		FRDGBufferUAVRef StatsBufferUAV = nullptr;

		TRDGUniformBufferRef<FNaniteRayTracingUniformParameters> UniformBuffer;

		const uint32 MaxReadbackBuffers = 4;

		ERayTracingMode bPrevMode = ERayTracingMode::Fallback;
		ERayTracingMode bCurrentMode = ERayTracingMode::Fallback;

		bool bUpdating = false;
		bool bInitialized = false;

#if !UE_BUILD_SHIPPING
		FDelegateHandle ScreenMessageDelegate;

		int32 NumVerticesHighWaterMark = 0;
		int32 NumIndicesHighWaterMark = 0;
		uint64 StagingBufferSizeHighWaterMark = 0;

		int32 NumVerticesHighWaterMarkPrev = 0;
		int32 NumIndicesHighWaterMarkPrev = 0;
		uint64 StagingBufferSizeHighWaterMarkPrev = 0;
#endif // UE_BUILD_SHIPPING
	};

	extern RENDERER_API TGlobalResource<FRayTracingManager> GRayTracingManager;

} // namespace Nanite

#endif // RHI_RAYTRACING
