// Copyright Epic Games, Inc. All Rights Reserved.

#include "NaniteRayTracing.h"

#if RHI_RAYTRACING

#include "Rendering/NaniteStreamingManager.h"

#include "NaniteStreamOut.h"
#include "NaniteSceneProxy.h"
#include "NaniteShared.h"

#include "ShaderPrintParameters.h"

#include "PrimitiveSceneInfo.h"
#include "ScenePrivate.h"
#include "SceneInterface.h"
#include "LogRenderer.h"

#include "RenderGraphUtils.h"

#include "RendererOnScreenNotification.h"

#include "NaniteRayTracingDefinitions.h"

/*
* TODO:
* - StagingAuxiliaryDataBuffer
*	- Keep track of how many pages/clusters are streamed-in per resource
*		and allocate less staging memory than the very conservative (Data.NumClusters * NANITE_MAX_CLUSTER_TRIANGLES)
* 
* - Defragment AuxiliaryDataBuffer
* 
* - VB/IB Buffers
*	- Resize VB/IB buffers dynamically instead of always allocating max size
*	- Store vertices and indices in the same buffer in a single allocation
* 
* - Support reserved resources to avoid copy when resizing auxiliary data buffer
*/

static bool GNaniteRayTracingUpdate = true;
static FAutoConsoleVariableRef CVarNaniteRayTracingUpdate(
	TEXT("r.RayTracing.Nanite.Update"),
	GNaniteRayTracingUpdate,
	TEXT("Whether to process Nanite RayTracing update requests."),
	ECVF_RenderThreadSafe
);

static bool GNaniteRayTracingForceUpdateVisible = false;
static FAutoConsoleVariableRef CVarNaniteRayTracingForceUpdateVisible(
	TEXT("r.RayTracing.Nanite.ForceUpdateVisible"),
	GNaniteRayTracingForceUpdateVisible,
	TEXT("Force BLAS of visible primitives to be updated next frame."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarNaniteRayTracingLodBias(
	TEXT("r.RayTracing.Nanite.LODBias"), 0.0f,
	TEXT("LOD bias for nanite geometry in ray tracing. 0 = full detail. >0 = reduced detail."),
	ECVF_RenderThreadSafe);

static float GNaniteRayTracingMinCutError = 0.0f;
static FAutoConsoleVariableRef CVarNaniteRayTracingMinCutError(
	TEXT("r.RayTracing.Nanite.MinCutError"),
	GNaniteRayTracingMinCutError,
	TEXT("Global minimum cut error to control quality when using Nanite Ray Tracing."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarNaniteRayTracingOffscreenLodBias(
	TEXT("r.RayTracing.Nanite.Offscreen.LODBias"), 1.0f,
	TEXT("LOD bias for offscreen nanite geometry in ray tracing. 0 = full detail. >0 = reduced detail."),
	ECVF_RenderThreadSafe);

static float GNaniteRayTracingOffscreenMinCutError = 4.0f;
static FAutoConsoleVariableRef CVarNaniteRayTracingOffscreenMinCutError(
	TEXT("r.RayTracing.Nanite.Offscreen.MinCutError"),
	GNaniteRayTracingOffscreenMinCutError,
	TEXT("Global target cut error when generating Nanite streaming requests for instances in ray tracing scene."),
	ECVF_RenderThreadSafe
);

static bool GNaniteRayTracingUseReferenceInstances = true;
static FAutoConsoleVariableRef CVarNaniteRayTracingUseReferenceInstances(
	TEXT("r.RayTracing.Nanite.UseReferenceInstances"),
	GNaniteRayTracingUseReferenceInstances,
	TEXT("Whether streaming requests and BLAS LOD should be based on a per-resource 'reference instance'."),
	ECVF_RenderThreadSafe
);

static bool GNaniteRayTracingBLASCache = true;
static FAutoConsoleVariableRef CVarNaniteRayTracingBLASCache(
	TEXT("r.RayTracing.Nanite.BLAS.Cache"),
	GNaniteRayTracingBLASCache,
	TEXT("Whether to enable the BLAS cache for Nanite Ray Tracing. Requires UseReferenceInstances."),
	ECVF_RenderThreadSafe
);

static float GNaniteRayTracingBLASCacheRelativeErrorTolerance = 0.5f;
static FAutoConsoleVariableRef CVarNaniteRayTracingBLASCacheRelativeErrorTolerance(
	TEXT("r.RayTracing.Nanite.BLAS.Cache.RelativeErrorTolerance"),
	GNaniteRayTracingBLASCacheRelativeErrorTolerance,
	TEXT("Relative error tolerance for BLAS cache hits (0 = exact match, >=1 frozen).\n")
	TEXT("Note that cluster streaming can still trigger BLAS rebuilds regardless of this tolerance."),
	ECVF_RenderThreadSafe
);

static int32 GNaniteRayTracingBLASCacheSizeMB = 64;
static FAutoConsoleVariableRef CVarNaniteRayTracingBLASCacheSizeMB(
	TEXT("r.RayTracing.Nanite.BLAS.Cache.Size"),
	GNaniteRayTracingBLASCacheSizeMB,
	TEXT("Size of the BLAS cache buffer in MB."),
	ECVF_RenderThreadSafe
);

static bool GNaniteRayTracingDriveStreaming = false;
static FAutoConsoleVariableRef CVarNaniteRayTracingDriveStreaming(
	TEXT("r.RayTracing.Nanite.DriveStreaming"),
	GNaniteRayTracingDriveStreaming,
	TEXT("Whether to drive Nanite streaming based on instances in ray tracing scene using Nanite Ray Tracing."),
	ECVF_RenderThreadSafe
);

static int32 GNaniteRayTracingMaxNumVertices = 16 * 1024 * 1024;
static FAutoConsoleVariableRef CVarNaniteRayTracingMaxNumVertices(
	TEXT("r.RayTracing.Nanite.StreamOut.MaxNumVertices"),
	GNaniteRayTracingMaxNumVertices,
	TEXT("Max number of vertices to stream out per frame."),
	ECVF_RenderThreadSafe
);

static int32 GNaniteRayTracingMaxNumIndices = 64 * 1024 * 1024;
static FAutoConsoleVariableRef CVarNaniteRayTracingMaxNumIndices(
	TEXT("r.RayTracing.Nanite.StreamOut.MaxNumIndices"),
	GNaniteRayTracingMaxNumIndices,
	TEXT("Max number of indices to stream out per frame."),
	ECVF_RenderThreadSafe
);

static int32 GNaniteRayTracingMaxBuiltPrimitivesPerFrame = 8 * 1024 * 1024;
static FAutoConsoleVariableRef CVarNaniteRayTracingMaxBuiltPrimitivesPerFrame(
	TEXT("r.RayTracing.Nanite.MaxBuiltPrimitivesPerFrame"),
	GNaniteRayTracingMaxBuiltPrimitivesPerFrame,
	TEXT("Limit number of BLAS built per frame based on a budget defined in terms of maximum number of triangles."),
	ECVF_RenderThreadSafe
);

static int32 GNaniteRayTracingMaxStagingBufferSizeMB = 1024;
static FAutoConsoleVariableRef CVarNaniteRayTracingMaxStagingBufferSizeMB(
	TEXT("r.RayTracing.Nanite.MaxStagingBufferSizeMB"),
	GNaniteRayTracingMaxStagingBufferSizeMB,
	TEXT("Limit the size of the staging buffer used during stream out (lower values can cause updates to be throttled)\n")
	TEXT("Default   = 1024 MB.\n")
	TEXT("Max value = 2048 MB."),
	ECVF_RenderThreadSafe
);

static int32 GNaniteRayTracingBLASScratchSizeMultipleMB = 64;
static FAutoConsoleVariableRef CVarNaniteRayTracingBLASScratchSizeMultipleMB(
	TEXT("r.RayTracing.Nanite.BLASScratchSizeMultipleMB"),
	GNaniteRayTracingBLASScratchSizeMultipleMB,
	TEXT("Round the size of the BLAS build scratch buffer to be a multiple of this value.\n")
	TEXT("This helps maintain consistent memory usage and prevent memory usage spikes.\n")
	TEXT("Default = 64 MB."),
	ECVF_RenderThreadSafe
);

static bool GNaniteRayTracingProfileStreamOut = false;
static FAutoConsoleVariableRef CVarNaniteRayTracingProfileStreamOut(
	TEXT("r.RayTracing.Nanite.ProfileStreamOut"),
	GNaniteRayTracingProfileStreamOut,
	TEXT("[Development only] Stream out pending requests every frame in order to measure performance."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarNaniteRayTracingCLASBufferSizeMB(
	TEXT("r.RayTracing.Nanite.CLASBufferSize"),
	512,
	TEXT("Limit the size of the CLAS Buffer\n")
	TEXT("Max value = 2048 MB."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarNaniteRayTracingCLASAllocationSize(
	TEXT("r.RayTracing.Nanite.CLASAllocationSize"),
	16384,
	TEXT("Size of allocated chunks in the CLAS buffer. CLASes will be greedily fit into these chunks.\n")
	TEXT("Clamped and aligned internally to minimum required CLAS size."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarNaniteRayTracingMaxPageInstallBatchSize(
	TEXT("r.RayTracing.Nanite.MaxPageInstallBatchSize"),
	32,
	TEXT("Max number of pages to install per batch (clamped to minimum 1).\n")
	TEXT("Higher allows more potential parallelism for the GPU, but requires (often significantly) more scratch VRAM space."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarNaniteRayTracingDebug(
	TEXT("r.RayTracing.Nanite.Debug"),
	false,
	TEXT("Whether to show Nanite Ray Tracing debug info."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarNaniteRayTracingDebugHistograms(
	TEXT("r.RayTracing.Nanite.Debug.Histograms"),
	false,
	TEXT("Whether to show size distribution histograms in debug output."),
	ECVF_RenderThreadSafe
);

DECLARE_GPU_STAT(NaniteRayTracingUpdateStreaming);
DECLARE_GPU_STAT(NaniteRayTracingRebuildBLAS);
DECLARE_GPU_STAT(NaniteRayTracingProcessBuildRequests);

DECLARE_STATS_GROUP(TEXT("Nanite RayTracing"), STATGROUP_NaniteRayTracing, STATCAT_Advanced);

DECLARE_DWORD_ACCUMULATOR_STAT(TEXT("In-flight Updates"), STAT_NaniteRayTracingInFlightUpdates, STATGROUP_NaniteRayTracing);
DECLARE_DWORD_COUNTER_STAT(TEXT("Stream Out Requests"), STAT_NaniteRayTracingStreamOutRequests, STATGROUP_NaniteRayTracing);
DECLARE_DWORD_COUNTER_STAT(TEXT("Failed Stream Out Requests"), STAT_NaniteRayTracingFailedStreamOutRequests, STATGROUP_NaniteRayTracing);
DECLARE_DWORD_COUNTER_STAT(TEXT("Scheduled Builds"), STAT_NaniteRayTracingScheduledBuilds, STATGROUP_NaniteRayTracing);
DECLARE_DWORD_COUNTER_STAT(TEXT("Scheduled Builds - Num Primitives"), STAT_NaniteRayTracingScheduledBuildsNumPrimitives, STATGROUP_NaniteRayTracing);
DECLARE_DWORD_ACCUMULATOR_STAT(TEXT("Pending Builds"), STAT_NaniteRayTracingPendingBuilds, STATGROUP_NaniteRayTracing);
DECLARE_MEMORY_STAT(TEXT("Auxiliary Data Buffer"), STAT_NaniteRayTracingAuxiliaryDataBuffer, STATGROUP_NaniteRayTracing);
DECLARE_MEMORY_STAT(TEXT("Staging Auxiliary Data Buffer"), STAT_NaniteRayTracingStagingAuxiliaryDataBuffer, STATGROUP_NaniteRayTracing);

DECLARE_MEMORY_STAT(TEXT("CLAS Allocated Size"), STAT_NaniteRayTracingCLASAllocatedSize, STATGROUP_NaniteRayTracing);
DECLARE_MEMORY_STAT(TEXT("CLAS Buffer Size"), STAT_NaniteRayTracingCLASBufferSize, STATGROUP_NaniteRayTracing);
DECLARE_MEMORY_STAT(TEXT("Staging Size"), STAT_NaniteRayTracingStagingSize, STATGROUP_NaniteRayTracing);
DECLARE_MEMORY_STAT(TEXT("Scratch Size"), STAT_NaniteRayTracingScratchSize, STATGROUP_NaniteRayTracing);
DECLARE_DWORD_COUNTER_STAT(TEXT("CLAS Builds"), STAT_NaniteRayTracingCLASBuilds, STATGROUP_NaniteRayTracing);

DECLARE_DWORD_COUNTER_STAT(TEXT("BLAS Builds"), STAT_NaniteRayTracingBLASBuilds, STATGROUP_NaniteRayTracing);

static const uint32 GMinAuxiliaryBufferEntries = 4 * 1024 * 1024; // buffer size will be 16MB
static const uint32 GDisabledMinAuxiliaryBufferEntries = 8; // used when Nanite Ray Tracing is not enabled

namespace Nanite
{
	// Helper for setting required permutations and parameters for debug stats
	template <typename ShaderType>
	static void SetStatsArgsAndPermutation(FRDGBufferUAVRef StatsBufferUAV, typename ShaderType::FParameters* OutPassParameters, typename ShaderType::FPermutationDomain& OutPermutationVector)
	{
		OutPassParameters->OutStatsBuffer = StatsBufferUAV;
		OutPermutationVector.template Set<typename ShaderType::FGenerateStatsDim>(StatsBufferUAV != nullptr);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FRayTracingQueueParameters, )
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FQueuePassState>, QueueState)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, Nodes)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, CandidateClusters)
		SHADER_PARAMETER(uint32, MaxNodes)
		SHADER_PARAMETER(uint32, MaxCandidateClusters)
	END_SHADER_PARAMETER_STRUCT()

	class FRayTracingStreamingInitQueueCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FRayTracingStreamingInitQueueCS);
		SHADER_USE_PARAMETER_STRUCT(FRayTracingStreamingInitQueueCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_STRUCT_INCLUDE(FRayTracingQueueParameters, QueueParameters)
			SHADER_PARAMETER_STRUCT_INCLUDE(FRayTracingLoadBalancer::FShaderParameters, LoadBalancerParameters)

			SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrint)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FStreamingTraversalRequest>, StreamingTraversalRequests)
			SHADER_PARAMETER(uint32, NumStreamingTraversalRequests)
		END_SHADER_PARAMETER_STRUCT()

		class FUseReferenceInstancesDim : SHADER_PERMUTATION_BOOL("USE_REFERENCE_INSTANCES");
		using FPermutationDomain = TShaderPermutationDomain<FUseReferenceInstancesDim>;

		static int32 GetThreadGroupSize(FPermutationDomain PermutationVector)
		{
			if (PermutationVector.Get<FUseReferenceInstancesDim>())
			{
				return 64;
			}
			else
			{
				return FRayTracingLoadBalancer::ThreadGroupSize;
			}
		}

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			FRayTracingLoadBalancer::SetShaderDefines(OutEnvironment);

			const FPermutationDomain PermutationVector(Parameters.PermutationId);
			OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetThreadGroupSize(PermutationVector));
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FRayTracingStreamingInitQueueCS, "/Engine/Private/Nanite/NaniteRayTracing.usf", "InitQueueCS", SF_Compute);

	struct FNaniteRayTracingStreamingTraversalCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingStreamingTraversalCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingStreamingTraversalCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)

			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, HierarchyBuffer)
			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, ClusterPageData)
			SHADER_PARAMETER(FIntVector4, PageConstants)

			SHADER_PARAMETER_STRUCT_INCLUDE(FRayTracingQueueParameters, QueueParameters)

			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CurrentNodeIndirectArgs)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, NextNodeIndirectArgs)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer, OutStreamingRequests)
			SHADER_PARAMETER(uint32, StreamingRequestsBufferVersion)
			SHADER_PARAMETER(uint32, StreamingRequestsBufferSize)

			SHADER_PARAMETER(uint32, RenderFlags)

			SHADER_PARAMETER(float, MinCutError)
			SHADER_PARAMETER(float, OffscreenMinCutError)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, ReferenceErrors)
			SHADER_PARAMETER(uint32, UseReferenceErrors)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FStreamingTraversalRequest>, StreamingTraversalRequests)
			SHADER_PARAMETER(uint32, NumStreamingTraversalRequests)

			SHADER_PARAMETER(uint32, NodeLevel)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FPackedNaniteView>, PackedNaniteViews)

			SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrint)

			RDG_BUFFER_ACCESS(IndirectArgs, ERHIAccess::IndirectArgs)
		END_SHADER_PARAMETER_STRUCT()

		class FCullingTypeDim : SHADER_PERMUTATION_SPARSE_INT("CULLING_TYPE", NANITE_CULLING_TYPE_NODES, NANITE_CULLING_TYPE_CLUSTERS); // don't need cluster permutation
		class FUseReferenceInstancesDim : SHADER_PERMUTATION_BOOL("USE_REFERENCE_INSTANCES");
		using FPermutationDomain = TShaderPermutationDomain<FCullingTypeDim, FUseReferenceInstancesDim>;

		static inline void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			OutEnvironment.SetDefine(TEXT("NANITE_HIERARCHY_TRAVERSAL"), 1);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingStreamingTraversalCS, "/Engine/Private/Nanite/NaniteRayTracing.usf", "NaniteRayTracingStreamingTraversalCS", SF_Compute);

	class FNaniteRayTracingCalculateReferenceErrorCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingCalculateReferenceErrorCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingCalculateReferenceErrorCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)
			SHADER_PARAMETER_STRUCT_INCLUDE(FRayTracingLoadBalancer::FShaderParameters, LoadBalancerParameters)
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FPackedNaniteView>, PackedNaniteViews)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWReferenceErrors)

			SHADER_PARAMETER(float, MinCutError)
			SHADER_PARAMETER(float, OffscreenMinCutError)
		END_SHADER_PARAMETER_STRUCT()

		using FPermutationDomain = TShaderPermutationDomain<>;

		static constexpr int32 ThreadGroupSize = FRayTracingLoadBalancer::ThreadGroupSize;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			FRayTracingLoadBalancer::SetShaderDefines(OutEnvironment);

			OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), ThreadGroupSize);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingCalculateReferenceErrorCS, "/Engine/Private/Nanite/NaniteRayTracing.usf", "CalculateReferenceErrorCS", SF_Compute);

	class FNaniteRayTracingStreamOutClusterCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingStreamOutClusterCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingStreamOutClusterCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FClusterStreamOutRequest>, StreamOutRequests)
			SHADER_PARAMETER(uint32, NumStreamOutRequests)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<GPU_VIRTUAL_ADDRESS>, ResourceAddresses)

			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, SegmentMappingBuffer)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWVertexBuffer)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWIndexBuffer)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWGeometryIndexAndFlagsBuffer)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWClusterCount)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<RAYTRACING_CLUSTER_OPS_BUILD_CLAS_DESC>, RWCLASBuildDescs)

			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, ClusterPageData)
			SHADER_PARAMETER(FIntVector4, PageConstants)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, OutStatsBuffer)
		END_SHADER_PARAMETER_STRUCT()

		class FGenerateStatsDim : SHADER_PERMUTATION_BOOL("NANITERT_GENERATE_STATS");
		using FPermutationDomain = TShaderPermutationDomain<FGenerateStatsDim>;

		static const uint32 ThreadGroupSize = 64;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), ThreadGroupSize);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingStreamOutClusterCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "StreamOutClusterCS", SF_Compute);

	class FNaniteRayTracingResetAllocatorCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingResetAllocatorCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingResetAllocatorCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<int>, RWAllocator)
			SHADER_PARAMETER(uint32, PoolSize)
		END_SHADER_PARAMETER_STRUCT()

		using FPermutationDomain = TShaderPermutationDomain<>;

		static const uint32 ThreadGroupSize = 64;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), ThreadGroupSize);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingResetAllocatorCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "ResetAllocatorCS", SF_Compute);

	class FNaniteRayTracingClampAllocatorCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingClampAllocatorCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingClampAllocatorCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<int>, RWAllocator)
			SHADER_PARAMETER(uint32, PoolSize)
		END_SHADER_PARAMETER_STRUCT()

		using FPermutationDomain = TShaderPermutationDomain<>;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingClampAllocatorCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "ClampAllocatorCS", SF_Compute);

	class FNaniteRayTracingAllocateClusterCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingAllocateClusterCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingAllocateClusterCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FAllocateClustersRequest>, AllocateClustersRequests)
			SHADER_PARAMETER(uint32, NumAllocateClustersRequests)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, CLASSizes)
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<GPU_VIRTUAL_ADDRESS>, CLASAddresses)
			SHADER_PARAMETER(uint32, NumCLAS)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<GPU_VIRTUAL_ADDRESS>, ResourceAddresses)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<int>, RWAllocator)
			SHADER_PARAMETER(uint32, AllocationSize)
			SHADER_PARAMETER(uint32, PoolSize)
			SHADER_PARAMETER(uint32, CLASBufferSize)
			SHADER_PARAMETER(uint32, CLASAlignment)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWMoveCount)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<GPU_VIRTUAL_ADDRESS>, RWSrcAddresses)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<GPU_VIRTUAL_ADDRESS>, RWDstAddresses)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWPageDataBuffer)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWCLASData)
		END_SHADER_PARAMETER_STRUCT()

		using FPermutationDomain = TShaderPermutationDomain<>;

		static const uint32 ThreadGroupSize = 64;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), ThreadGroupSize);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingAllocateClusterCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "AllocateClusterCS", SF_Compute);

	class FNaniteRayTracingFreeClusterCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingFreeClusterCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingFreeClusterCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FFreeClustersRequest>, FreeClustersRequests)
			SHADER_PARAMETER(uint32, NumFreeClustersRequests)

			SHADER_PARAMETER(uint32, NumCLAS)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<int>, RWAllocator)
			SHADER_PARAMETER(uint32, AllocationSize)
			SHADER_PARAMETER(uint32, PoolSize)
			SHADER_PARAMETER(uint32, CLASBufferSize)
			SHADER_PARAMETER(uint32, CLASAlignment)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWPageDataBuffer)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWCLASData)
		END_SHADER_PARAMETER_STRUCT()

		using FPermutationDomain = TShaderPermutationDomain<>;

		static const uint32 ThreadGroupSize = 64;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), ThreadGroupSize);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingFreeClusterCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "FreeClusterCS", SF_Compute);

	class FRayTracingGeometryClusterCutInitQueueCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FRayTracingGeometryClusterCutInitQueueCS);
		SHADER_USE_PARAMETER_STRUCT(FRayTracingGeometryClusterCutInitQueueCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_STRUCT_INCLUDE(FRayTracingQueueParameters, QueueParameters)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FGeometryClusterCutRequest>, CutRequests)
			SHADER_PARAMETER(uint32, NumCutRequests)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<GPU_VIRTUAL_ADDRESS>, ResourceAddressesBuffer)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<RAYTRACING_CLUSTER_OPS_BUILD_BLAS_DESC>, RWBLASBuildBuffer)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FNaniteRayTracingASCacheEntry>, BLASCacheAllocationTable)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteRayTracingASCacheMetadata>, RWBLASCacheMetadata)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWBLASData)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWBLASCountBuffer)
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, ReferenceErrors)
			SHADER_PARAMETER(uint32, BLASCacheEnabled)
			SHADER_PARAMETER(float, CacheRelativeErrorTolerance)

			SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrint)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, OutStatsBuffer)
		END_SHADER_PARAMETER_STRUCT()

		class FGenerateStatsDim : SHADER_PERMUTATION_BOOL("NANITERT_GENERATE_STATS");
		using FPermutationDomain = TShaderPermutationDomain<FGenerateStatsDim>;

		static const uint32 ThreadGroupSize = 64;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), ThreadGroupSize);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FRayTracingGeometryClusterCutInitQueueCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "InitQueueCS", SF_Compute);

	struct FNaniteRayTracingGeometryClusterCutTraversalCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingGeometryClusterCutTraversalCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingGeometryClusterCutTraversalCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)

			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, HierarchyBuffer)
			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, ClusterPageData)
			SHADER_PARAMETER(FIntVector4, PageConstants)

			SHADER_PARAMETER_STRUCT_INCLUDE(FRayTracingQueueParameters, QueueParameters)

			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CurrentNodeIndirectArgs)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, NextNodeIndirectArgs)

			SHADER_PARAMETER(float, MinCutError)
			SHADER_PARAMETER(uint32, NodeLevel)

			SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrint)

			RDG_BUFFER_ACCESS(IndirectArgs, ERHIAccess::IndirectArgs)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FGeometryClusterCutRequest>, CutRequests)
			SHADER_PARAMETER(uint32, NumCutRequests)

			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, PageDataBuffer)
			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, CLASDataBuffer)
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, ReferenceErrors)
			SHADER_PARAMETER(uint32, UseReferenceErrors)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<RAYTRACING_CLUSTER_OPS_BUILD_BLAS_DESC>, RWBLASBuildDescs)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<GPU_VIRTUAL_ADDRESS>, RWCLASAddresses)

			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, BLASDataBuffer)
		END_SHADER_PARAMETER_STRUCT()

		class FCullingTypeDim : SHADER_PERMUTATION_SPARSE_INT("CULLING_TYPE", NANITE_CULLING_TYPE_NODES, NANITE_CULLING_TYPE_CLUSTERS); // don't need cluster permutation
		using FPermutationDomain = TShaderPermutationDomain<FCullingTypeDim>;

		static inline void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			OutEnvironment.SetDefine(TEXT("NANITE_HIERARCHY_TRAVERSAL"), 1);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingGeometryClusterCutTraversalCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "HierarchyTraversalCS", SF_Compute);

	class FNaniteRayTracingAllocateBLASCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingAllocateBLASCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingAllocateBLASCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, BLASSizes)
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<GPU_VIRTUAL_ADDRESS>, BLASAddresses)
			SHADER_PARAMETER(uint32, NumBLAS)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<GPU_VIRTUAL_ADDRESS>, ResourceAddresses)

			SHADER_PARAMETER(uint32, BLASAlignment)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWMoveCount)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<GPU_VIRTUAL_ADDRESS>, RWSrcAddresses)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<GPU_VIRTUAL_ADDRESS>, RWDstAddresses)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FGeometryClusterCutRequest>, CutRequests)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWBLASData)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, RWBLASCacheRequest)
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FNaniteRayTracingASCacheEntry>, BLASCacheAllocationTable)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteRayTracingASCacheMetadata>, RWBLASCacheMetadata)
			SHADER_PARAMETER(uint32, BLASCacheEnabled)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, OutStatsBuffer)
		END_SHADER_PARAMETER_STRUCT()

		class FGenerateStatsDim : SHADER_PERMUTATION_BOOL("NANITERT_GENERATE_STATS");
		using FPermutationDomain = TShaderPermutationDomain<FGenerateStatsDim>;

		static const uint32 ThreadGroupSize = 64;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), ThreadGroupSize);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingAllocateBLASCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "AllocateBLASCS", SF_Compute);

	class FNaniteRayTracingDebugCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingDebugCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingDebugCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrint)

			SHADER_PARAMETER(uint32, CLASPoolAllocationSize)
			SHADER_PARAMETER(uint32, CLASPoolSize)
			SHADER_PARAMETER(uint32, CLASBufferSize)
			SHADER_PARAMETER(uint32, NumCLAS)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<int>, CLASAllocator)
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, CLASHistogram)
			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, CLASSizeSum)

			SHADER_PARAMETER(uint32, BLASCacheBufferSize)
			SHADER_PARAMETER(uint32, BLASStagingBufferSize)
			SHADER_PARAMETER(uint32, NumBLAS)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, BLASHistogram)

			SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, StatsBuffer)

			SHADER_PARAMETER(uint32, ShowHistograms)
		END_SHADER_PARAMETER_STRUCT()

		using FPermutationDomain = TShaderPermutationDomain<>;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingDebugCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "DebugCS", SF_Compute);

	class FNaniteRayTracingDebugHistogramCS : public FNaniteGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FNaniteRayTracingDebugHistogramCS);
		SHADER_USE_PARAMETER_STRUCT(FNaniteRayTracingDebugHistogramCS, FNaniteGlobalShader);

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, ASDataBuffer)
			SHADER_PARAMETER(uint32, NumAS)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWHistogram)
			SHADER_PARAMETER(FVector2f, HistogramScaleBias)
			SHADER_PARAMETER(uint32, HistogramSize)

			SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, RWSizeSum)
		END_SHADER_PARAMETER_STRUCT()

		using FPermutationDomain = TShaderPermutationDomain<>;

		static const uint32 ThreadGroupSize = 64;

		static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);

			OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), ThreadGroupSize);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			if (!FDataDrivenShaderPlatformInfo::GetSupportsRayTracingClusterOps(Parameters.Platform))
			{
				return false;
			}

			return FNaniteGlobalShader::ShouldCompilePermutation(Parameters);
		}
	};
	IMPLEMENT_GLOBAL_SHADER(FNaniteRayTracingDebugHistogramCS, "/Engine/Private/Nanite/NaniteClusterRayTracing.usf", "DebugHistogramCS", SF_Compute);

	/*
	* Stream out nanite mesh data into buffers in a uncompressed format
	*/
	void FRayTracingManager::StreamOutClusters(
		FRDGBuilder& GraphBuilder,
		FGlobalShaderMap* ShaderMap,
		uint32 NumRequests,
		FRDGBufferRef RequestBuffer,
		FRDGBufferRef ResourceAddressesBuffer,
		FRDGBufferRef SegmentMappingBufferRDG)
	{
		check(StreamOutBuffers);

		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(StreamOutBuffers->ClusterCountBuffer), 0);

		FNaniteRayTracingStreamOutClusterCS::FParameters* PassParams = GraphBuilder.AllocParameters<FNaniteRayTracingStreamOutClusterCS::FParameters>();
		PassParams->StreamOutRequests = GraphBuilder.CreateSRV(RequestBuffer);
		PassParams->NumStreamOutRequests = NumRequests;

		PassParams->ResourceAddresses = GraphBuilder.CreateSRV(ResourceAddressesBuffer);
		PassParams->SegmentMappingBuffer = GraphBuilder.CreateSRV(SegmentMappingBufferRDG);

		PassParams->RWVertexBuffer = GraphBuilder.CreateUAV(StreamOutBuffers->ClusterVertexBuffer);
		PassParams->RWIndexBuffer = GraphBuilder.CreateUAV(StreamOutBuffers->ClusterIndexBuffer);
		PassParams->RWGeometryIndexAndFlagsBuffer = GraphBuilder.CreateUAV(StreamOutBuffers->ClusterGeometryIndexAndFlagsBuffer);

		PassParams->RWClusterCount = GraphBuilder.CreateUAV(StreamOutBuffers->ClusterCountBuffer);
		PassParams->RWCLASBuildDescs = GraphBuilder.CreateUAV(StreamOutBuffers->ClusterBuildDescBuffer);

		PassParams->ClusterPageData = Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
		PassParams->PageConstants.X = 0;
		PassParams->PageConstants.Y = Nanite::GStreamingManager.GetMaxStreamingPages();

		FNaniteRayTracingStreamOutClusterCS::FPermutationDomain PermutationVector;
		SetStatsArgsAndPermutation<FNaniteRayTracingStreamOutClusterCS>(StatsBufferUAV, PassParams, PermutationVector);

		auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingStreamOutClusterCS>(PermutationVector);
		FIntVector GroupCount = FComputeShaderUtils::GetGroupCountWrapped(NumRequests, FNaniteRayTracingStreamOutClusterCS::ThreadGroupSize);

		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::StreamOutClusters"), ComputeShader, PassParams, GroupCount);
	}

	static FRDGBufferRef ResizeBufferIfNeeded(FRDGBuilder& GraphBuilder, TRefCountPtr<FRDGPooledBuffer>& ExternalBuffer, FRDGBufferDesc BufferDesc, const TCHAR* Name, bool bCopy, EAllowShrinking AllowShrinking)
	{
		if (!ExternalBuffer)
		{
			FRDGBuffer* InternalBufferNew = GraphBuilder.CreateBuffer(BufferDesc, Name);
			ExternalBuffer = GraphBuilder.ConvertToExternalBuffer(InternalBufferNew);
			return InternalBufferNew;
		}

		FRDGBufferRef BufferRDG = GraphBuilder.RegisterExternalBuffer(ExternalBuffer);

		if (BufferDesc.GetSize() > BufferRDG->GetSize()) // grow
		{
			FRDGBufferRef SrcBufferRDG = BufferRDG;

			BufferRDG = GraphBuilder.CreateBuffer(BufferDesc, Name);
			ExternalBuffer = GraphBuilder.ConvertToExternalBuffer(BufferRDG);

			if (bCopy)
			{
				AddCopyBufferPass(GraphBuilder, BufferRDG, SrcBufferRDG);
			}
		}
		else if (AllowShrinking == EAllowShrinking::Yes && BufferDesc.GetSize() < BufferRDG->GetSize() / 2) // shrink
		{
			FRDGBufferRef SrcBufferRDG = BufferRDG;

			BufferRDG = GraphBuilder.CreateBuffer(BufferDesc, Name);
			ExternalBuffer = GraphBuilder.ConvertToExternalBuffer(BufferRDG);

			if (bCopy)
			{
				AddCopyBufferPass(GraphBuilder, BufferRDG, 0, SrcBufferRDG, 0, BufferDesc.GetSize());
			}
		}

		return BufferRDG;
	}

	static FRDGBufferRef ResizeStructuredBufferIfNeeded(FRDGBuilder& GraphBuilder, TRefCountPtr<FRDGPooledBuffer>& ExternalBuffer, uint32 BytesPerElement, uint32 NumElements, const TCHAR* Name, bool bCopy, EAllowShrinking AllowShrinking)
	{
		return ResizeBufferIfNeeded(GraphBuilder, ExternalBuffer, FRDGBufferDesc::CreateStructuredDesc(BytesPerElement, NumElements), Name, bCopy, AllowShrinking);
	}

	static FRDGBufferRef ResizeByteAddressBufferIfNeeded(FRDGBuilder& GraphBuilder, TRefCountPtr<FRDGPooledBuffer>& ExternalBuffer, uint32 ByteAddressBufferSize, const TCHAR* Name, bool bCopy, EAllowShrinking AllowShrinking)
	{
		return ResizeBufferIfNeeded(GraphBuilder, ExternalBuffer, FRDGBufferDesc::CreateByteAddressDesc(ByteAddressBufferSize), Name, bCopy, AllowShrinking);
	}

	static uint32 GetAuxiliaryEntrySize()
	{
		return NaniteAssembliesSupported() ? sizeof(FUintVector2) : sizeof(uint32);
	}

	static uint32 CalculateAuxiliaryDataSizeInUints(uint32 NumTriangles)
	{
		return NumTriangles; // (one uint per triangle)
	}

	FRayTracingManager::FRayTracingManager()
	{

	}

	FRayTracingManager::~FRayTracingManager()
	{

	}

	void FRayTracingManager::Initialize()
	{
#if !UE_BUILD_SHIPPING
		ScreenMessageDelegate = FRendererOnScreenNotification::Get().AddLambda([this](TMultiMap<FCoreDelegates::EOnScreenMessageSeverity, FText >& OutMessages)
			{
				if (NumVerticesHighWaterMark >= GNaniteRayTracingMaxNumVertices)
				{
					OutMessages.Add(FCoreDelegates::EOnScreenMessageSeverity::Warning, FText::FromString(FString::Printf(TEXT("Nanite Ray Tracing vertex buffer overflow detected, increase 'r.RayTracing.Nanite.StreamOut.MaxNumVertices' to avoid rendering artifacts."))));
					OutMessages.Add(FCoreDelegates::EOnScreenMessageSeverity::Warning, FText::FromString(FString::Printf(TEXT(" Required max num vertices for update = %d, currently = %d"), NumVerticesHighWaterMark, GNaniteRayTracingMaxNumVertices)));
					if (NumVerticesHighWaterMark > NumVerticesHighWaterMarkPrev)
					{
						UE_LOGF(LogRenderer, Warning, "Nanite Ray Tracing vertex buffer overflow detected, increase 'r.RayTracing.Nanite.StreamOut.MaxNumVertices' to avoid rendering artifacts.\n"
							" Required max num vertices for update = %d, currently = %d", NumVerticesHighWaterMark, GNaniteRayTracingMaxNumVertices);
						NumVerticesHighWaterMarkPrev = NumVerticesHighWaterMark;
					}
				}

				if (NumIndicesHighWaterMark >= GNaniteRayTracingMaxNumIndices)
				{
					OutMessages.Add(FCoreDelegates::EOnScreenMessageSeverity::Warning, FText::FromString(FString::Printf(TEXT("Nanite Ray Tracing index buffer overflow detected, increase 'r.RayTracing.Nanite.StreamOut.MaxNumIndices' to avoid rendering artifacts."))));
					OutMessages.Add(FCoreDelegates::EOnScreenMessageSeverity::Warning, FText::FromString(FString::Printf(TEXT(" Required max num indices for update = %d, currently = %d"), NumIndicesHighWaterMark, GNaniteRayTracingMaxNumIndices)));
					if (NumIndicesHighWaterMark > NumIndicesHighWaterMarkPrev)
					{
						UE_LOGF(LogRenderer, Warning, "Nanite Ray Tracing index buffer overflow detected, increase 'r.RayTracing.Nanite.StreamOut.MaxNumIndices' to avoid rendering artifacts.\n"
							" Required max num indices for update = %d, currently = %d", NumIndicesHighWaterMark, GNaniteRayTracingMaxNumIndices);
						NumIndicesHighWaterMarkPrev = NumIndicesHighWaterMark;
					}
				}

				if (StagingBufferSizeHighWaterMark >= GNaniteRayTracingMaxStagingBufferSizeMB * (1024ull * 1024ull))
				{
					OutMessages.Add(FCoreDelegates::EOnScreenMessageSeverity::Warning, FText::FromString(FString::Printf(TEXT("Nanite Ray Tracing staging buffer overflow detected, increase 'r.RayTracing.Nanite.MaxStagingBufferSizeMB' to avoid rendering artifacts."))));
					OutMessages.Add(FCoreDelegates::EOnScreenMessageSeverity::Warning, FText::FromString(FString::Printf(TEXT(" Required for update = %lld, currently = %d"), StagingBufferSizeHighWaterMark / (1024ull * 1024ull), GNaniteRayTracingMaxStagingBufferSizeMB)));
					if (StagingBufferSizeHighWaterMark > StagingBufferSizeHighWaterMarkPrev)
					{
						UE_LOGF(LogRenderer, Warning, "Nanite Ray Tracing staging buffer overflow detected, increase 'r.RayTracing.Nanite.MaxStagingBufferSizeMB' to avoid rendering artifacts.\n"
							" Required for update = %lld, currently = %d", StagingBufferSizeHighWaterMark / (1024ull * 1024ull), GNaniteRayTracingMaxStagingBufferSizeMB);
						StagingBufferSizeHighWaterMarkPrev = StagingBufferSizeHighWaterMark;
					}
				}

				if (bCLASBufferFull)
				{
					OutMessages.Add(FCoreDelegates::EOnScreenMessageSeverity::Warning, FText::FromString(FString::Printf(TEXT("Nanite Ray Tracing CLAS Buffer overflow detected, increase 'r.RayTracing.Nanite.CLASBufferSize' to avoid rendering artifacts."))));
				}

			});
#endif
	}

	void FRayTracingManager::Shutdown()
	{
#if !UE_BUILD_SHIPPING
		FRendererOnScreenNotification::Get().Remove(ScreenMessageDelegate);
#endif
	}

	void FRayTracingManager::InitRHI(FRHICommandListBase&)
	{
		AuxiliaryDataBuffer = AllocatePooledBuffer(FRDGBufferDesc::CreateByteAddressDesc(GetAuxiliaryEntrySize() * GDisabledMinAuxiliaryBufferEntries), TEXT("NaniteRayTracing.AuxiliaryDataBuffer"));
		SET_MEMORY_STAT(STAT_NaniteRayTracingAuxiliaryDataBuffer, AuxiliaryDataBuffer->GetSize());

		SegmentMappingBuffer = AllocatePooledBuffer(FRDGBufferDesc::CreateByteAddressDesc(sizeof(uint32) * 8), TEXT("NaniteRayTracing.SegmentMappingBuffer"));

		if (!DoesPlatformSupportNanite(GMaxRHIShaderPlatform))
		{
			return;
		}

		if (GRHIGlobals.RayTracing.SupportsClusterOps)
		{
			// Fill out driver queries
			if (CLASMaxSize <= 0)
			{
				FRayTracingClusterOperationInitializer ClusterOpInitializer = {};
				ClusterOpInitializer.MaxResultCount = 1;
				ClusterOpInitializer.Type = ERayTracingClusterOperationType::CLAS_BUILD;
				ClusterOpInitializer.Mode = ERayTracingClusterOperationMode::EXPLICIT_DESTINATIONS;
				ClusterOpInitializer.Flags = ERayTracingClusterOperationFlags::FAST_TRACE;
				ClusterOpInitializer.Operation.CLAS.VertexFormat = VET_Float3;
				ClusterOpInitializer.Operation.CLAS.MaxGeometryIndex = 1;
				ClusterOpInitializer.Operation.CLAS.MaxUniqueGeometryCount = 1;
				ClusterOpInitializer.Operation.CLAS.MaxTriangleCount = NANITE_MAX_CLUSTER_TRIANGLES;
				ClusterOpInitializer.Operation.CLAS.MaxVertexCount = NANITE_MAX_CLUSTER_VERTICES;
				ClusterOpInitializer.Operation.CLAS.MaxTotalTriangleCount = NANITE_MAX_CLUSTER_TRIANGLES;
				ClusterOpInitializer.Operation.CLAS.MaxTotalVertexCount = NANITE_MAX_CLUSTER_VERTICES;

				FRayTracingClusterOperationSize CLASSize = RHICalcRayTracingClusterOperationSize(ClusterOpInitializer);
				CLASMaxSize = CLASSize.ResultMaxSizeInBytes;
			}

			{
				FRDGBufferDesc ResourceAddressesBufferDesc = FRDGBufferDesc::CreateStructuredUploadDesc(sizeof(GPU_VIRTUAL_ADDRESS), 4);
				ResourceAddressesBufferDesc.Usage |= EBufferUsageFlags::MultiGPUAllocate | EBufferUsageFlags::Volatile;
				CLASResourceAddressesBuffer = AllocatePooledBuffer(ResourceAddressesBufferDesc, TEXT("NaniteRayTracing.CLASResourceAddressesBuffer"));
			}
			{
				FRDGBufferDesc ResourceAddressesBufferDesc = FRDGBufferDesc::CreateStructuredUploadDesc(sizeof(GPU_VIRTUAL_ADDRESS), 3);
				ResourceAddressesBufferDesc.Usage |= EBufferUsageFlags::MultiGPUAllocate | EBufferUsageFlags::Volatile;
				BLASResourceAddressesBuffer = AllocatePooledBuffer(ResourceAddressesBufferDesc, TEXT("NaniteRayTracing.BLASResourceAddressesBuffer"));
			}

			CLASReadbackDatas.SetNum(MaxReadbackBuffers);

			for (auto& ReadbackData : CLASReadbackDatas)
			{
				ReadbackData.ReadbackBuffer = new FRHIGPUBufferReadback(TEXT("NaniteRayTracing.CLASReadbackBuffer"));
			}

		}

		ReadbackBuffers.SetNum(MaxReadbackBuffers);

		for (auto& ReadbackData : ReadbackBuffers)
		{
			ReadbackData.MeshDataReadbackBuffer = new FRHIGPUBufferReadback(TEXT("NaniteRayTracing.MeshDataReadbackBuffer"));
		}
		
		bInitialized = true;
	}

	void FRayTracingManager::ReleaseRHI()
	{
		AuxiliaryDataBuffer.SafeRelease();
		SegmentMappingBuffer.SafeRelease();

		if (!bInitialized)
		{
			return;
		}

		bInitialized = false;

		VertexBuffer.SafeRelease();
		IndexBuffer.SafeRelease();

		for (auto& ReadbackData : ReadbackBuffers)
		{
			delete ReadbackData.MeshDataReadbackBuffer;
			ReadbackData.MeshDataReadbackBuffer = nullptr;
		}

		ReadbackBuffers.Empty();

		for (auto& ReadbackData : CLASReadbackDatas)
		{
			delete ReadbackData.ReadbackBuffer;
			ReadbackData.ReadbackBuffer = nullptr;
		}

		CLASReadbackDatas.Empty();

		CLASResourceAddressesBuffer.SafeRelease();
		BLASResourceAddressesBuffer.SafeRelease();

		StagingAuxiliaryDataBuffer.SafeRelease();		
	}

	void FRayTracingManager::Add(FPrimitiveSceneInfo* SceneInfo)
	{
		if (!IsRayTracingEnabled() || (GetRayTracingMode() == ERayTracingMode::Fallback))
		{
			return;
		}

		auto NaniteProxy = static_cast<Nanite::FSceneProxyBase*>(SceneInfo->Proxy);
		Nanite::FResourcePrimitiveInfo PrimitiveInfo = NaniteProxy->GetResourcePrimitiveInfo();

		// TODO: Should use both RuntimeResourceID and HierarchyOffset as identifier for raytracing geometry
		// For example, FNaniteGeometryCollectionSceneProxy can use the same RuntimeResourceID with different HierarchyOffsets
		// (FNaniteGeometryCollectionSceneProxy are not supported in raytracing yet)
		uint32& Id = ResourceToRayTracingIdMap.FindOrAdd(PrimitiveInfo.ResourceID, INDEX_NONE);

		FInternalData* Data;

		if (Id == INDEX_NONE)
		{
			Nanite::FResourceMeshInfo MeshInfo = NaniteProxy->GetResourceMeshInfo();
			check(MeshInfo.NumClusters);

			Data = new FInternalData;

			Data->RuntimeResourceID = PrimitiveInfo.ResourceID;
			Data->HierarchyOffset = PrimitiveInfo.HierarchyOffset;
			Data->NumClusters = MeshInfo.NumClusters;
			Data->NumNodes = MeshInfo.NumNodes;
			Data->NumVertices = MeshInfo.NumVertices;
			Data->NumTriangles = MeshInfo.NumTriangles;
			Data->NumMaterials = MeshInfo.NumMaterials;
			Data->NumSegments = MeshInfo.NumSegments;
			Data->SegmentMapping = MeshInfo.SegmentMapping;
			Data->SegmentMappingOffset = SegmentMappingAllocator.Allocate(Data->SegmentMapping.Num());
			Data->bAssembly = MeshInfo.bAssembly;
			Data->DebugName = MeshInfo.DebugName;

			Data->NumResidentClusters = 0;
			Data->NumResidentClustersUpdate = MeshInfo.NumResidentClusters;

			Data->PrimitiveId = INDEX_NONE;

			Id = Geometries.Add(Data);

			if (Data->NumResidentClustersUpdate > 0)
			{
				// some clusters are already streamed in and RequestUpdates(...) is only called when new pages are streamed in/out
				// so request an update here to make sure we build ray tracing geometry with the currently available data
				UpdateRequests.Add(Id);
			}

			PendingSegmentMappingUpload.Add(Id);
			PendingSegmentMappingUploadCount += Data->SegmentMapping.Num();
		}
		else
		{
			Data = Geometries[Id];
		}

		Data->Primitives.Add(SceneInfo);

		PendingRemoves.Remove(Id);

		NaniteProxy->SetRayTracingId(Id);
		NaniteProxy->SetRayTracingDataOffset(Data->AuxiliaryDataOffset);

		// Mask off the version bits (high bits above NANITE_MAX_GPU_PAGES_BITS) so per-resource buffers are
		// sized by the root-page index space rather than the unbounded versioned RuntimeResourceID range.
		MaxRootPageIndex = FMath::Max(MaxRootPageIndex, PrimitiveInfo.ResourceID & NANITE_MAX_GPU_PAGES_MASK);
	}

	void FRayTracingManager::Remove(FPrimitiveSceneInfo* SceneInfo)
	{
		if (!IsRayTracingAllowed())
		{
			return;
		}

		auto NaniteProxy = static_cast<Nanite::FSceneProxyBase*>(SceneInfo->Proxy);

		const uint32 GeometryId = NaniteProxy->GetRayTracingId();

		if(GeometryId == INDEX_NONE)
		{
			check(NaniteProxy->GetRayTracingDataOffset() == INDEX_NONE);
			return;
		}

		FInternalData* Data = Geometries[GeometryId];

		Data->Primitives.Remove(SceneInfo);
		if (Data->Primitives.IsEmpty())
		{
			PendingRemoves.Add(GeometryId);
		}

		NaniteProxy->SetRayTracingId(INDEX_NONE);
		NaniteProxy->SetRayTracingDataOffset(INDEX_NONE);
	}

	void FRayTracingManager::RequestUpdates(const TMap<uint32, uint32>& InUpdateRequests, const TSet<FPageInfo>& InInstalledPages, const TSet<FPageInfo>& InUninstalledPages)
	{
		for (auto& PageInfo : InUninstalledPages)
		{
			PendingCLASBuildPages.Remove(PageInfo);

			if (PendingInstalledPages.Remove(PageInfo) > 0)
			{
				continue;
			}

			PendingUninstalledPages.Add(PageInfo);
		}

		PendingInstalledPages.Append(InInstalledPages);

		if (!IsRayTracingEnabled() || (GetRayTracingMode() == ERayTracingMode::Fallback))
		{
			return;
		}

		for (auto& Elem : InUpdateRequests)
		{
			uint32 RuntimeResourceID = Elem.Key;
			uint32* GeometryId = ResourceToRayTracingIdMap.Find(RuntimeResourceID);

			if (GeometryId != nullptr)
			{
				FInternalData& Data = *Geometries[*GeometryId];
				Data.NumResidentClustersUpdate = Elem.Value;
				check(Data.NumResidentClustersUpdate > 0);

				UpdateRequests.Add(*GeometryId);
			}
		}
	}

	void FRayTracingManager::AddVisiblePrimitive(const FPrimitiveSceneInfo* SceneInfo)
	{
		check(GetRayTracingMode() != ERayTracingMode::Fallback);

		auto NaniteProxy = static_cast<Nanite::FSceneProxyBase*>(SceneInfo->Proxy);

		const uint32 Id = NaniteProxy->GetRayTracingId();
		check(Id != INDEX_NONE);

		FInternalData* Data = Geometries[Id];
		Data->PrimitiveId = SceneInfo->GetPersistentIndex().Index;

		VisibleGeometries.Add(Id);

		VisiblePrimitives.Add(SceneInfo);
	}

	void AddPass_InitNodeCullArgs(FRDGBuilder& GraphBuilder, FGlobalShaderMap* ShaderMap, FRDGEventName&& PassName, FRDGBufferUAVRef QueueState, FRDGBufferRef NodeCullArgs0, FRDGBufferRef NodeCullArgs1, uint32 CullingPass);
	void AddPass_InitClusterCullArgs(FRDGBuilder& GraphBuilder, FGlobalShaderMap* ShaderMap, FRDGEventName&& PassName, FRDGBufferUAVRef QueueState, FRDGBufferRef ClusterCullArgs, uint32 CullingPass);

	static void AddInitQueuePass(
		FRDGBuilder& GraphBuilder,
		FGlobalShaderMap* ShaderMap,
		FRayTracingQueueParameters& QueueParameters,
		const FRayTracingLoadBalancer::FGPUData& LoadBalancerGPUData,
		bool bUseReferenceErrors,
		FRDGBufferSRVRef RequestSRV,
		uint32 NumRequests)
	{
		// Reset queue to empty state
		AddClearUAVPass(GraphBuilder, QueueParameters.QueueState, 0);

		// Init queue with requests
		{
			FRayTracingStreamingInitQueueCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FRayTracingStreamingInitQueueCS::FParameters>();
			PassParameters->QueueParameters = QueueParameters;
			PassParameters->StreamingTraversalRequests = RequestSRV;
			PassParameters->NumStreamingTraversalRequests = NumRequests;

			ShaderPrint::SetParameters(GraphBuilder, PassParameters->ShaderPrint);

			FRayTracingStreamingInitQueueCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FRayTracingStreamingInitQueueCS::FUseReferenceInstancesDim>(bUseReferenceErrors);

			auto ComputeShader = ShaderMap->GetShader<FRayTracingStreamingInitQueueCS>(PermutationVector);

			if (bUseReferenceErrors)
			{
				FIntVector GroupCount = FComputeShaderUtils::GetGroupCountWrapped(NumRequests, FRayTracingStreamingInitQueueCS::GetThreadGroupSize(PermutationVector));
				FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::StreamingInitQueue"), ComputeShader, PassParameters, GroupCount);
			}
			else
			{
				LoadBalancerGPUData.AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::StreamingInitQueue"), ComputeShader, PassParameters);
			}
		}
	}

	static void GetQueueParams(FRDGBuilder& GraphBuilder, FGlobalShaderMap* ShaderMap, bool bStreamingRequestsOnly, FRayTracingQueueParameters& OutQueueParameters)
	{
		const uint32 MaxNodes = Nanite::FGlobalResources::GetMaxNodes();
		const uint32 MaxCandidateClusters = bStreamingRequestsOnly ? 0 : Nanite::FGlobalResources::GetMaxCandidateClusters();

		FRDGBufferRef QueueState = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32) + 2 * (6 * sizeof(uint32)), 1), TEXT("NaniteRayTracing.QueueState"));

		// Allocate buffer for nodes
		FRDGBufferRef NodesBuffer = nullptr;
		{
			const uint32 CandidateNodeSizeInUints = bStreamingRequestsOnly ? 2 : 3;
			FRDGBufferDesc Desc = FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), MaxNodes * CandidateNodeSizeInUints);
			Desc.Usage = EBufferUsageFlags(Desc.Usage | BUF_ByteAddressBuffer);
			NodesBuffer = GraphBuilder.CreateBuffer(Desc, TEXT("NaniteRayTracing.NodesBuffer"));
		}

		// Allocate candidate cluster buffer
		FRDGBufferRef CandidateClustersBuffer = nullptr;
		if (MaxCandidateClusters > 0)
		{
			const uint32 CandidateClusterSizeInUints = 3;
			FRDGBufferDesc Desc = FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), MaxCandidateClusters * CandidateClusterSizeInUints);
			Desc.Usage = EBufferUsageFlags(Desc.Usage | BUF_ByteAddressBuffer);
			CandidateClustersBuffer = GraphBuilder.CreateBuffer(Desc, TEXT("NaniteStreamOut.CandidateClustersBuffer"));
		}

		OutQueueParameters.QueueState = GraphBuilder.CreateUAV(QueueState);
		OutQueueParameters.Nodes = GraphBuilder.CreateUAV(NodesBuffer);
		OutQueueParameters.CandidateClusters = CandidateClustersBuffer ? GraphBuilder.CreateUAV(CandidateClustersBuffer) : nullptr;
		OutQueueParameters.MaxNodes = MaxNodes;
		OutQueueParameters.MaxCandidateClusters = MaxCandidateClusters;
	}

	static void AddPass_StreamingTraversal(
		FRDGBuilder& GraphBuilder,
		FGlobalShaderMap* ShaderMap,
		FSceneUniformBuffer& SceneUniformBuffer,
		const FRayTracingLoadBalancer::FGPUData& LoadBalancerGPUData,
		FRDGBufferSRVRef PackedNaniteViews,
		bool bUseReferenceErrors,
		FRDGBufferSRVRef ReferenceErrorsSRV,
		FRDGBufferSRVRef RequestSRV,
		uint32 NumRequests,
		FRayTracingQueueParameters& QueueParameters
	)
	{
		AddInitQueuePass(
			GraphBuilder,
			ShaderMap,
			QueueParameters,
			LoadBalancerGPUData,
			bUseReferenceErrors,
			RequestSRV,
			NumRequests);

		FNaniteRayTracingStreamingTraversalCS::FParameters SharedParameters;

		SharedParameters.Scene = SceneUniformBuffer.GetBuffer(GraphBuilder);

		SharedParameters.QueueParameters = QueueParameters;

		SharedParameters.HierarchyBuffer = Nanite::GStreamingManager.GetHierarchySRV(GraphBuilder);
		SharedParameters.ClusterPageData = Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
		SharedParameters.PageConstants.X = 0;
		SharedParameters.PageConstants.Y = Nanite::GStreamingManager.GetMaxStreamingPages();

		SharedParameters.MinCutError = GNaniteRayTracingMinCutError;
		SharedParameters.OffscreenMinCutError = GNaniteRayTracingOffscreenMinCutError;

		SharedParameters.ReferenceErrors = ReferenceErrorsSRV;
		SharedParameters.UseReferenceErrors = bUseReferenceErrors;

		SharedParameters.StreamingTraversalRequests = RequestSRV;
		SharedParameters.NumStreamingTraversalRequests = NumRequests;

		FRDGBufferRef StreamingRequests = Nanite::GStreamingManager.GetStreamingRequestsBuffer(GraphBuilder);

		SharedParameters.OutStreamingRequests = GraphBuilder.CreateUAV(StreamingRequests);
		SharedParameters.StreamingRequestsBufferVersion = GStreamingManager.GetStreamingRequestsBufferVersion();
		SharedParameters.StreamingRequestsBufferSize = StreamingRequests->Desc.NumElements;

		SharedParameters.RenderFlags = 0;
		SharedParameters.RenderFlags |= NANITE_RENDER_FLAG_OUTPUT_STREAMING_REQUESTS;

		ShaderPrint::SetParameters(GraphBuilder, SharedParameters.ShaderPrint);

		FNaniteRayTracingStreamingTraversalCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FNaniteRayTracingStreamingTraversalCS::FUseReferenceInstancesDim>(bUseReferenceErrors);

		{
			RDG_EVENT_SCOPE(GraphBuilder, "StreamingTraversal");

			// Node passes
			{
				FRDGBufferRef NodeCullArgs0 = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc((NANITE_MAX_CLUSTER_HIERARCHY_DEPTH + 1) * NANITE_NODE_CULLING_ARG_COUNT), TEXT("Nanite.CullArgs0"));
				FRDGBufferRef NodeCullArgs1 = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc((NANITE_MAX_CLUSTER_HIERARCHY_DEPTH + 1) * NANITE_NODE_CULLING_ARG_COUNT), TEXT("Nanite.CullArgs1"));

				AddPass_InitNodeCullArgs(GraphBuilder, ShaderMap, RDG_EVENT_NAME("InitNodeCullArgs"), QueueParameters.QueueState, NodeCullArgs0, NodeCullArgs1, 0);

				PermutationVector.Set<FNaniteRayTracingStreamingTraversalCS::FCullingTypeDim>(NANITE_CULLING_TYPE_NODES);
				auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingStreamingTraversalCS>(PermutationVector);

				const uint32 MaxLevels = Nanite::GStreamingManager.GetMaxHierarchyLevels();
				for (uint32 NodeLevel = 0; NodeLevel < MaxLevels; NodeLevel++)
				{
					auto* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingStreamingTraversalCS::FParameters>(&SharedParameters);

					FRDGBufferRef CurrentIndirectArgs = (NodeLevel & 1) ? NodeCullArgs1 : NodeCullArgs0;
					FRDGBufferRef NextIndirectArgs = (NodeLevel & 1) ? NodeCullArgs0 : NodeCullArgs1;

					PassParameters->CurrentNodeIndirectArgs = GraphBuilder.CreateSRV(CurrentIndirectArgs);
					PassParameters->NextNodeIndirectArgs = GraphBuilder.CreateUAV(NextIndirectArgs);
					PassParameters->IndirectArgs = CurrentIndirectArgs;
					PassParameters->NodeLevel = NodeLevel;
					PassParameters->PackedNaniteViews = PackedNaniteViews;

					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("NodeCull_%d", NodeLevel),
						ComputeShader,
						PassParameters,
						CurrentIndirectArgs,
						NodeLevel * NANITE_NODE_CULLING_ARG_COUNT * sizeof(uint32)
					);
				}
			}
		}
	}
	
	void FRayTracingManager::UpdateStreaming(FRDGBuilder& GraphBuilder, TConstArrayView<FViewInfo> Views, FSceneUniformBuffer& SceneUniformBuffer, FIntPoint RasterTextureSize)
	{
		if ((!GNaniteRayTracingDriveStreaming && !GNaniteRayTracingUseReferenceInstances) || Views.IsEmpty() || VisiblePrimitives.IsEmpty())
		{
			return;
		}

		RDG_EVENT_SCOPE_STAT(GraphBuilder, NaniteRayTracingUpdateStreaming, "NaniteRayTracing::UpdateStreaming");

		//ue-todo: Refactor to use EShaderPlatform directly instead of converting from FeatureLevel
		FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GShaderPlatformForFeatureLevel[GetFeatureLevel()]);

		// TODO: MaxPixelsPerEdgeMultipler should match rasterization + LOD bias (ie: FDeferredShadingSceneRenderer::RenderNanite(...))

		float LODScaleFactor = FMath::Exp2(-CVarNaniteRayTracingLodBias.GetValueOnRenderThread());
		float LODScaleFactorOffscreen = FMath::Exp2(-CVarNaniteRayTracingOffscreenLodBias.GetValueOnRenderThread());

		LODScaleFactor *= Nanite::GStreamingManager.GetQualityScaleFactor();
		LODScaleFactorOffscreen *= Nanite::GStreamingManager.GetQualityScaleFactor();

		float MaxPixelsPerEdgeMultipler = 1.0f / LODScaleFactor;
		float MaxPixelsPerEdgeMultiplerOffscreen = 1.0f / LODScaleFactorOffscreen;

		FRDGUploadData<Nanite::FPackedView> PackedViews(GraphBuilder, Views.Num() * 2);

		for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
		{
			PackedViews[ViewIndex * 2 + 0] = Nanite::CreatePackedViewFromViewInfo(
				Views[ViewIndex],
				RasterTextureSize,
				NANITE_VIEW_FLAG_NEAR_CLIP, // TODO: HZB test
				/* StreamingPriorityCategory = */ 2,
				/* MinBoundsRadius = */ 0.0f,
				MaxPixelsPerEdgeMultipler,
				nullptr // TODO: HZB test
			);

			PackedViews[ViewIndex * 2 + 1] = Nanite::CreatePackedViewFromViewInfo(
				Views[ViewIndex],
				RasterTextureSize,
				NANITE_VIEW_FLAG_NEAR_CLIP,
				/* StreamingPriorityCategory = */ 0,
				/* MinBoundsRadius = */ 0.0f,
				MaxPixelsPerEdgeMultiplerOffscreen,
				nullptr
			);
		}

		FRDGBufferRef PackedNaniteViewsBuffer = CreateStructuredBuffer(GraphBuilder, TEXT("NaniteRayTracing.Views"), PackedViews);
		FRDGBufferSRVRef PackedNaniteViewsSRV = GraphBuilder.CreateSRV(PackedNaniteViewsBuffer);

		FRayTracingLoadBalancer LoadBalancer;

		// TODO: move this to tasks
		for (const FPrimitiveSceneInfo* SceneInfo : VisiblePrimitives)
		{
			const int32 InstanceSceneDataOffset = SceneInfo->GetInstanceSceneDataOffset();
			const int32 NumInstanceSceneDataEntries = SceneInfo->GetNumInstanceSceneDataEntries();

			if (NumInstanceSceneDataEntries > 0u)
			{
				LoadBalancer.Add(InstanceSceneDataOffset, NumInstanceSceneDataEntries, SceneInfo->GetPersistentIndex().Index);
			}
		}

		FRayTracingLoadBalancer::FGPUData LoadBalancerGPUData = LoadBalancer.Upload(GraphBuilder);
		
		if (GNaniteRayTracingUseReferenceInstances)
		{
			CalculateReferenceErrors(GraphBuilder, SceneUniformBuffer, PackedNaniteViewsSRV, LoadBalancerGPUData);
		}
		else
		{
			ReferenceErrorsBuffer = nullptr;
		}

		if (GNaniteRayTracingDriveStreaming)
		{
			FRayTracingQueueParameters QueueParameters;
			GetQueueParams(GraphBuilder, ShaderMap, /*bStreamingRequestsOnly*/ true, QueueParameters);

			const bool bUseReferenceErrors = ReferenceErrorsBuffer != nullptr;
			FRDGBufferRef ReferenceErrorsBufferRDG = bUseReferenceErrors ? GraphBuilder.RegisterExternalBuffer(ReferenceErrorsBuffer) : GSystemTextures.GetDefaultStructuredBuffer(GraphBuilder, sizeof(uint32));

			uint32 NumRequests = 0;
			FRDGBufferRef RequestBuffer = GSystemTextures.GetDefaultStructuredBuffer(GraphBuilder, sizeof(FStreamingTraversalRequest));
			
			if (bUseReferenceErrors)
			{
				FRDGUploadData<FStreamingTraversalRequest> UploadData(GraphBuilder, Geometries.Num());

				for (FInternalData* Data : Geometries)
				{
					FStreamingTraversalRequest& Request = UploadData[NumRequests];
					Request.HierarchyOffset = Data->HierarchyOffset;
					Request.RuntimeResourceID = Data->RuntimeResourceID;

					++NumRequests;
				}

				RequestBuffer = CreateStructuredBuffer(GraphBuilder, TEXT("NaniteRayTracing.GeometryClusterCutRequestBuffer"), UploadData);
			}

			AddPass_StreamingTraversal(
				GraphBuilder,
				ShaderMap,
				SceneUniformBuffer,
				LoadBalancerGPUData,
				PackedNaniteViewsSRV,
				bUseReferenceErrors,
				GraphBuilder.CreateSRV(ReferenceErrorsBufferRDG),
				GraphBuilder.CreateSRV(RequestBuffer),
				NumRequests,
				QueueParameters);
		}
	}

	void FRayTracingManager::CalculateReferenceErrors(
		FRDGBuilder& GraphBuilder,
		FSceneUniformBuffer& SceneUniformBuffer,
		FRDGBufferSRVRef PackedNaniteViews,
		const FRayTracingLoadBalancer::FGPUData& LoadBalancerGPUData)
	{
		FRDGBufferRef ReferenceErrorsBufferRDG = ResizeStructuredBufferIfNeeded(GraphBuilder, ReferenceErrorsBuffer, sizeof(uint32), MaxRootPageIndex + 1, TEXT("NaniteRayTracing.ReferenceErrorsBuffer"), /*bCopy*/ false, EAllowShrinking::Yes);
		FRDGBufferUAVRef ReferenceErrorsBufferUAV = GraphBuilder.CreateUAV(ReferenceErrorsBufferRDG);

		//ue-todo: Refactor to use EShaderPlatform directly instead of converting from FeatureLevel
		FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GShaderPlatformForFeatureLevel[GetFeatureLevel()]);

		AddClearUAVPass(GraphBuilder, ReferenceErrorsBufferUAV, UINT32_MAX);

		{
			FNaniteRayTracingCalculateReferenceErrorCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingCalculateReferenceErrorCS::FParameters>();
			PassParameters->Scene = SceneUniformBuffer.GetBuffer(GraphBuilder);
			PassParameters->PackedNaniteViews = PackedNaniteViews;
			PassParameters->MinCutError = GNaniteRayTracingMinCutError;
			PassParameters->OffscreenMinCutError = GNaniteRayTracingOffscreenMinCutError;
			PassParameters->RWReferenceErrors = ReferenceErrorsBufferUAV;

			FNaniteRayTracingCalculateReferenceErrorCS::FPermutationDomain PermutationVector;

			auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingCalculateReferenceErrorCS>(PermutationVector);

			LoadBalancerGPUData.AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::CalculateReferenceError"), ComputeShader, PassParameters);
		}
	}

	void FRayTracingManager::ProcessUpdateRequests(FRDGBuilder& GraphBuilder, FSceneUniformBuffer& SceneUniformBuffer)
	{
		// D3D12 limits resources to 2048MB.
		GNaniteRayTracingMaxStagingBufferSizeMB = FMath::Min(GNaniteRayTracingMaxStagingBufferSizeMB, 2048);

		if (!GNaniteRayTracingUpdate)
		{
			// TODO: shrink staging buffer and other unused resources?
			return;
		}

		ProcessPendingSegmentMappingUploads(GraphBuilder);
		ProcessUpdateRequestsModeStreamOut(GraphBuilder, SceneUniformBuffer);
		ProcessUpdateRequestsModeCLAS(GraphBuilder, SceneUniformBuffer);
	}

	void FRayTracingManager::ProcessPendingSegmentMappingUploads(FRDGBuilder& GraphBuilder)
	{
		if (PendingSegmentMappingUpload.IsEmpty())
		{
			return;
		}

		SegmentMappingUploadBuffer.Init(GraphBuilder, PendingSegmentMappingUploadCount, sizeof(uint32), false, TEXT("NaniteRayTracing.SegmentMappingUploadBuffer"));

		for (uint32 GeometryId : PendingSegmentMappingUpload)
		{
			FInternalData& Data = *Geometries[GeometryId];

			for (int32 SegmentIndex = 0; SegmentIndex < Data.SegmentMapping.Num(); ++SegmentIndex)
			{
				SegmentMappingUploadBuffer.Add(Data.SegmentMappingOffset + SegmentIndex, &Data.SegmentMapping[SegmentIndex]);
			}
		}

		// TODO: Replace RoundUpToPowerOfTwo
		FRDGBufferRef SegmentMappingBufferRDG = ResizeByteAddressBufferIfNeeded(GraphBuilder, SegmentMappingBuffer, sizeof(uint32) * SegmentMappingAllocator.GetMaxSize(), TEXT("NaniteRayTracing.SegmentMappingBuffer"), /*bCopy*/ true, EAllowShrinking::No);

		SegmentMappingUploadBuffer.ResourceUploadTo(GraphBuilder, SegmentMappingBufferRDG);

		PendingSegmentMappingUpload.Empty();
		PendingSegmentMappingUploadCount = 0;
	}

	void FRayTracingManager::ProcessUpdateRequestsModeStreamOut(FRDGBuilder& GraphBuilder, FSceneUniformBuffer &SceneUniformBuffer)
	{
		if (GNaniteRayTracingForceUpdateVisible)
		{
			UpdateRequests.Append(VisibleGeometries);
			GNaniteRayTracingForceUpdateVisible = false;
		}

		if (GetRayTracingMode() != ERayTracingMode::StreamOut || bUpdating || UpdateRequests.IsEmpty())
		{
			// TODO: shrink staging buffer and other unused resources?
			return;
		}

		TSet<uint32> ToUpdate;

		uint32 NumMeshDataEntries = 0;
		uint32 NumAuxiliaryDataEntries = 0;

		const uint64 AuxiliaryEntrySize = GetAuxiliaryEntrySize();

		for (uint32 GeometryId : VisibleGeometries)
		{
			if (UpdateRequests.Contains(GeometryId))
			{
				FInternalData& Data = *Geometries[GeometryId];

				check(Data.NumResidentClustersUpdate > 0);
 				//check(Data.NumResidentClustersUpdate <= Data.NumClusters); // Temporary workaround: NumClusters from cooked data is not always correct for Geometry Collections: UE-194917

				// TODO: Investigate a more conservative MaxNumTriangles for assemblies
				const uint32 MaxNumTriangles = Data.bAssembly ? Data.NumTriangles : (Data.NumResidentClustersUpdate * NANITE_MAX_CLUSTER_TRIANGLES);
				const uint64 MaxNumAuxiliaryDataEntries = CalculateAuxiliaryDataSizeInUints(MaxNumTriangles);
				const uint64 NewNumAuxiliaryDataEntries = NumAuxiliaryDataEntries + MaxNumAuxiliaryDataEntries;
				const uint64 NewAuxiliaryDataBufferSize = NewNumAuxiliaryDataEntries * AuxiliaryEntrySize;

#if !UE_BUILD_SHIPPING
				StagingBufferSizeHighWaterMark = FMath::Max(StagingBufferSizeHighWaterMark, MaxNumAuxiliaryDataEntries * AuxiliaryEntrySize);
#endif

				if (NewAuxiliaryDataBufferSize >= (uint64)GNaniteRayTracingMaxStagingBufferSizeMB * (1024ull * 1024ull))
				{
					break;
				}

				check(NewAuxiliaryDataBufferSize <= (1u << 31)); // D3D12 limits resources to 2048MB.

				if (!GNaniteRayTracingProfileStreamOut) // don't remove request when profiling stream out
				{
					UpdateRequests.Remove(GeometryId);
				}
				ToUpdate.Add(GeometryId);

				Data.NumResidentClusters = Data.NumResidentClustersUpdate;

				check(!Data.bUpdating);
				Data.bUpdating = true;

				check(Data.BaseMeshDataOffset == -1);
				Data.BaseMeshDataOffset = NumMeshDataEntries;

				check(Data.StagingAuxiliaryDataOffset == INDEX_NONE);
				Data.StagingAuxiliaryDataOffset = NumAuxiliaryDataEntries;

				NumMeshDataEntries += (sizeof(FStreamOutMeshDataHeader) + sizeof(FStreamOutMeshDataSegment) * Data.NumSegments);
				NumAuxiliaryDataEntries = NewNumAuxiliaryDataEntries;
			}
		}

		if (ToUpdate.IsEmpty())
		{
			return;
		}

		RDG_EVENT_SCOPE(GraphBuilder, "Nanite::FRayTracingManager::ProcessUpdateRequests");

		bUpdating = true;

		FReadbackData& ReadbackData = ReadbackBuffers[ReadbackBuffersWriteIndex];
		check(ReadbackData.EntryGeometryId.IsEmpty());

		// Upload geometry data
		FRDGBufferRef RequestBuffer = nullptr;
		FRDGBufferRef SegmentMappingBufferRDG = GraphBuilder.RegisterExternalBuffer(SegmentMappingBuffer);
		
		{
			FRDGUploadData<FStreamOutRequest> UploadData(GraphBuilder, ToUpdate.Num());

			uint32 Index = 0;

			for (auto GeometryId : ToUpdate)
			{
				const FInternalData& Data = *Geometries[GeometryId];

				FStreamOutRequest& Request = UploadData[Index];
				Request.PrimitiveId = Data.PrimitiveId;
				Request.NumMaterials = Data.NumMaterials;
				Request.NumSegments = Data.NumSegments;
				Request.SegmentMappingOffset = Data.SegmentMappingOffset;
				Request.AuxiliaryDataOffset = Data.StagingAuxiliaryDataOffset;
				Request.MeshDataOffset = Data.BaseMeshDataOffset;

				ReadbackData.EntryGeometryId.Add(GeometryId);

				++Index;
			}

			INC_DWORD_STAT_BY(STAT_NaniteRayTracingInFlightUpdates, ToUpdate.Num());

			RequestBuffer = CreateStructuredBuffer(GraphBuilder, TEXT("NaniteRayTracing.RequestBuffer"), UploadData);
		}

		FRDGBufferDesc MeshDataBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), FMath::Max(NumMeshDataEntries, 32U));
		MeshDataBufferDesc.Usage |= BUF_SourceCopy;

		FRDGBufferRef MeshDataBuffer = GraphBuilder.CreateBuffer(MeshDataBufferDesc, TEXT("NaniteRayTracing.MeshDataBuffer"));

		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(MeshDataBuffer), 0);

		FRDGBufferRef StagingAuxiliaryDataBufferRDG;

		{
			const uint32 BufferNumAuxiliaryDataEntries = FMath::Max(NumAuxiliaryDataEntries, GMinAuxiliaryBufferEntries);
			const bool bCopy = false;
			StagingAuxiliaryDataBufferRDG = ResizeByteAddressBufferIfNeeded(GraphBuilder, StagingAuxiliaryDataBuffer, AuxiliaryEntrySize * BufferNumAuxiliaryDataEntries, TEXT("NaniteRayTracing.StagingAuxiliaryDataBuffer"), bCopy, EAllowShrinking::Yes);

			SET_MEMORY_STAT(STAT_NaniteRayTracingStagingAuxiliaryDataBuffer, StagingAuxiliaryDataBufferRDG->GetSize());
		}
		
		FRDGBufferRef VertexBufferRDG = ResizeStructuredBufferIfNeeded(GraphBuilder, VertexBuffer, sizeof(float), GNaniteRayTracingMaxNumVertices * 3, TEXT("NaniteRayTracing.VertexBuffer"), /*bCopy*/ false, EAllowShrinking::Yes);
		FRDGBufferRef IndexBufferRDG = ResizeStructuredBufferIfNeeded(GraphBuilder, IndexBuffer, sizeof(uint32), GNaniteRayTracingMaxNumIndices, TEXT("NaniteRayTracing.IndexBuffer"), /*bCopy*/ false, EAllowShrinking::Yes);

		StreamOutData(
			GraphBuilder,
			//ue-todo: Refactor to use EShaderPlatform directly instead of converting from FeatureLevel
			GetGlobalShaderMap(GShaderPlatformForFeatureLevel[GetFeatureLevel()]),
			SceneUniformBuffer,
			GNaniteRayTracingMinCutError,
			ToUpdate.Num(),
			RequestBuffer,
			SegmentMappingBufferRDG,
			MeshDataBuffer,
			StagingAuxiliaryDataBufferRDG,
			VertexBufferRDG,
			GNaniteRayTracingMaxNumVertices,
			IndexBufferRDG,
			GNaniteRayTracingMaxNumIndices);

		INC_DWORD_STAT_BY(STAT_NaniteRayTracingStreamOutRequests, ToUpdate.Num());

		if (!GNaniteRayTracingProfileStreamOut)
		{
			// readback
			{
				AddReadbackBufferPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::Readback"), MeshDataBuffer,
					[MeshDataReadbackBuffer = ReadbackData.MeshDataReadbackBuffer, MeshDataBuffer](FRDGAsyncTask, FRHICommandList& RHICmdList)
					{
						MeshDataReadbackBuffer->EnqueueCopy(RHICmdList, MeshDataBuffer->GetRHI(), 0u);
					});

				ReadbackData.NumMeshDataEntries = NumMeshDataEntries;

				ReadbackBuffersWriteIndex = (ReadbackBuffersWriteIndex + 1u) % MaxReadbackBuffers;
				ReadbackBuffersNumPending = FMath::Min(ReadbackBuffersNumPending + 1u, MaxReadbackBuffers);
			}
		}
		else
		{
			// if running profile mode, clear state for next frame

			bUpdating = false;

			for (auto GeometryId : ToUpdate)
			{
				FInternalData& Data = *Geometries[GeometryId];
				Data.bUpdating = false;
				Data.BaseMeshDataOffset = -1;
				Data.StagingAuxiliaryDataOffset = INDEX_NONE;
			}

			ReadbackData.EntryGeometryId.Empty();
		}

		ToUpdate.Empty();
	}

	void FRayTracingManager::ProcessUpdateRequestsModeCLAS(FRDGBuilder& GraphBuilder, FSceneUniformBuffer& SceneUniformBuffer)
	{
		if (GetRayTracingMode() == ERayTracingMode::CLAS)
		{
			RDG_EVENT_SCOPE(GraphBuilder, "NaniteRayTracing::UpdateModeCLAS");

			// Allocate buffers for the worst case cluster counts we may need
			const uint32 MaxClustersPerBuild = GetMaxPageInstallBatchSize() * NANITE_MAX_CLUSTERS_PER_PAGE;
			check(MaxClustersPerBuild > 0);
			if (!StreamOutBuffers || StreamOutBuffers->MaxClusters != MaxClustersPerBuild)
			{
				StreamOutBuffers = MakeShared<FClusterStreamOutBuffers>((uint32)MaxClustersPerBuild);
			}
			StreamOutBuffers->Initialize(GraphBuilder);

			if (CVarNaniteRayTracingDebug.GetValueOnRenderThread())
			{
				StatsBufferRDG = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), NANITERT_STAT_NUM), TEXT("NaniteRayTracing.StatsBuffer"));
				AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(StatsBufferRDG), 0);
				StatsBufferUAV = GraphBuilder.CreateUAV(StatsBufferRDG, ERDGUnorderedAccessViewFlags::SkipBarrier);
			}

			ResizeCLASBufferIfNeeded(GraphBuilder);
			ProcessPendingPages(GraphBuilder);
			ResizeBLASCacheIfNeeded(GraphBuilder);
			if (BLASCache)
			{
				BLASCache->ProcessReadbacks();
			}
			UpdateBLAS(GraphBuilder);
			DebugModeCLAS(GraphBuilder);

			StatsBufferRDG = nullptr;
			StatsBufferUAV = nullptr;
			StreamOutBuffers->Cleanup();
		}
		else
		{
			ResetPageInstallationState();

			CLASBuffer = nullptr;
			CLASAllocatorBuffer = nullptr;

			PageDataBuffer = nullptr;
			PageDataUploadBuffer.Release();

			CLASDataBuffer = nullptr;
			CLASDataAllocator.Empty();

			StreamOutBuffers = nullptr;

			CLASPoolAllocationSize = 0;
			CLASPoolSize = 0;
			CLASBufferAllocatedSize = 0;
			bCLASBufferFull = false;

			BLASStagingBuffer = nullptr;

			BLASDataBuffer = nullptr;

			BLASCache = nullptr;

			MaxGPUPageIndex = 0;

			SET_MEMORY_STAT(STAT_NaniteRayTracingCLASBufferSize, 0);
			SET_MEMORY_STAT(STAT_NaniteRayTracingStagingSize, 0);
			SET_MEMORY_STAT(STAT_NaniteRayTracingScratchSize, 0);
		}

		SET_MEMORY_STAT(STAT_NaniteRayTracingCLASAllocatedSize, CLASBufferAllocatedSize);
	}

	void FRayTracingManager::Update(FRHICommandListBase& RHICmdList)
	{
		const bool bUsingNaniteRayTracing = GetRayTracingMode() != ERayTracingMode::Fallback;

		if (!bUsingNaniteRayTracing && !bUpdating)
		{
			StagingAuxiliaryDataBuffer.SafeRelease();
			SET_MEMORY_STAT(STAT_NaniteRayTracingStagingAuxiliaryDataBuffer, 0);

			SegmentMappingBuffer.SafeRelease();

			VertexBuffer.SafeRelease();
			IndexBuffer.SafeRelease();

#if !UE_BUILD_SHIPPING
			NumVerticesHighWaterMark = 0;
			NumIndicesHighWaterMark = 0;
			StagingBufferSizeHighWaterMark = 0;
#endif
		}

		// process PendingRemoves
		{
			TSet<uint32> StillPendingRemoves;

			for (uint32 GeometryId : PendingRemoves)
			{
				FInternalData* Data = Geometries[GeometryId];

				if (Data->bUpdating)
				{
					// can't remove until update is finished, delay to next frame
					StillPendingRemoves.Add(GeometryId);
				}
				else
				{
					if (Data->AuxiliaryDataOffset != INDEX_NONE)
					{
						AuxiliaryDataAllocator.Free(Data->AuxiliaryDataOffset, Data->AuxiliaryDataSize);
					}

					if (Data->SegmentMappingOffset != INDEX_NONE)
					{
						SegmentMappingAllocator.Free(Data->SegmentMappingOffset, Data->SegmentMapping.Num());
					}

					if (PendingSegmentMappingUpload.Remove(GeometryId) > 0)
					{
						PendingSegmentMappingUploadCount -= Data->SegmentMapping.Num();
					}

					ResourceToRayTracingIdMap.Remove(Data->RuntimeResourceID);
					Geometries.RemoveAt(GeometryId);
					delete (Data);
				}
			}

			Swap(PendingRemoves, StillPendingRemoves);
		}

		const uint32 PrevScheduledBuildsNumPrimitives = ScheduledBuildsNumPrimitives;

		// scheduling pending builds
		{
			const uint32 PrevNumScheduled = ScheduledBuilds.Num();
			
			for (const FPendingBuild& PendingBuild : PendingBuilds)
			{
				if (ScheduledBuildsNumPrimitives >= GNaniteRayTracingMaxBuiltPrimitivesPerFrame)
				{
					break;
				}

				FInternalData& Data = *Geometries[PendingBuild.GeometryId];
				Data.RayTracingGeometryRHI = PendingBuild.RayTracingGeometryRHI;

				const FRayTracingGeometryInitializer& Initializer = Data.RayTracingGeometryRHI->GetInitializer();

				ScheduledBuildsNumPrimitives += Initializer.TotalPrimitiveCount;

				if (Data.AuxiliaryDataOffset != INDEX_NONE)
				{
					AuxiliaryDataAllocator.Free(Data.AuxiliaryDataOffset, Data.AuxiliaryDataSize);
				}
				Data.AuxiliaryDataSize = Initializer.TotalPrimitiveCount;
				Data.AuxiliaryDataOffset = AuxiliaryDataAllocator.Allocate(Data.AuxiliaryDataSize);

				for (auto& Primitive : Data.Primitives)
				{
					if (bUsingNaniteRayTracing)
					{
						Primitive->SetCachedRayTracingInstanceGeometryRHI(Data.RayTracingGeometryRHI, Data.NumSegments);
					}

					auto NaniteProxy = static_cast<Nanite::FSceneProxyBase*>(Primitive->Proxy);
					NaniteProxy->SetRayTracingDataOffset(Data.AuxiliaryDataOffset);

					Primitive->Scene->GPUScene.AddPrimitiveToUpdate(Primitive->GetPersistentIndex(), EPrimitiveDirtyState::ChangedOther);
				}

				ScheduledBuilds.Add(PendingBuild.GeometryId);
			}

			// not using RemoveAtSwap to avoid starving requests in the middle
			// not expecting significant number of elements remaining anyway
			PendingBuilds.RemoveAt(0, ScheduledBuilds.Num() - PrevNumScheduled);

			DEC_DWORD_STAT_BY(STAT_NaniteRayTracingPendingBuilds, ScheduledBuilds.Num() - PrevNumScheduled);
		}

		while (ReadbackBuffersNumPending > 0)
		{
			uint32 Index = (ReadbackBuffersWriteIndex + MaxReadbackBuffers - ReadbackBuffersNumPending) % MaxReadbackBuffers;
			FReadbackData& ReadbackData = ReadbackBuffers[Index];
			if (ReadbackData.MeshDataReadbackBuffer->IsReady())
			{
				ReadbackBuffersNumPending--;

				auto MeshDataReadbackBufferPtr = (const uint32*)ReadbackData.MeshDataReadbackBuffer->Lock(ReadbackData.NumMeshDataEntries * sizeof(uint32));

				for (int32 GeometryIndex = 0; GeometryIndex < ReadbackData.EntryGeometryId.Num(); ++GeometryIndex)
				{
					const uint32 GeometryId = ReadbackData.EntryGeometryId[GeometryIndex];
					FInternalData& Data = *Geometries[GeometryId];

					auto Header = (const FStreamOutMeshDataHeader*)(MeshDataReadbackBufferPtr + Data.BaseMeshDataOffset);
					auto Segments = (const FStreamOutMeshDataSegment*)(Header + 1);

					if (!Data.bAssembly)
					{
						check(Header->NumClusters <= Data.NumResidentClusters);
					}

					const uint32 VertexBufferOffset = Header->VertexBufferOffset;
					const uint32 IndexBufferOffset = Header->IndexBufferOffset;
					const uint32 NumVertices = Header->NumVertices;

					if (VertexBufferOffset == 0xFFFFFFFFu || IndexBufferOffset == 0xFFFFFFFFu)
					{
						// ran out of space in StreamOut buffers
						Data.bUpdating = false;
						Data.BaseMeshDataOffset = -1;

						check(Data.StagingAuxiliaryDataOffset != INDEX_NONE);
						Data.StagingAuxiliaryDataOffset = INDEX_NONE;

						UpdateRequests.Add(GeometryId); // request update again

						DEC_DWORD_STAT_BY(STAT_NaniteRayTracingInFlightUpdates, 1);
						INC_DWORD_STAT_BY(STAT_NaniteRayTracingFailedStreamOutRequests, 1);

#if !UE_BUILD_SHIPPING
						NumVerticesHighWaterMark = FMath::Max(NumVerticesHighWaterMark, (int32)Header->NumVertices);
						NumIndicesHighWaterMark = FMath::Max(NumIndicesHighWaterMark, (int32)Header->NumIndices);
#endif

						continue;
					}

					FRayTracingGeometryInitializer Initializer;
					Initializer.DebugName = Data.DebugName;
// 					Initializer.bFastBuild = false;
// 					Initializer.bAllowUpdate = false;
					Initializer.bAllowCompaction = false;

					Initializer.IndexBuffer = IndexBuffer->GetRHI();
					Initializer.IndexBufferStride = sizeof(uint32);
					Initializer.IndexBufferOffset = IndexBufferOffset * sizeof(uint32);

					Initializer.TotalPrimitiveCount = 0;

					Initializer.Segments.SetNum(Data.NumSegments);

					for (uint32 SegmentIndex = 0; SegmentIndex < Data.NumSegments; ++SegmentIndex)
					{
						const uint32 NumIndices = Segments[SegmentIndex].NumIndices;
						const uint32 FirstIndex = Segments[SegmentIndex].FirstIndex;

						FRayTracingGeometrySegment& Segment = Initializer.Segments[SegmentIndex];
						Segment.FirstPrimitive = FirstIndex / 3;
						Segment.NumPrimitives = NumIndices / 3;
						Segment.VertexBuffer = VertexBuffer->GetRHI();
						Segment.VertexBufferOffset = VertexBufferOffset * sizeof(FVector3f);
						Segment.MaxVertices = NumVertices;

						Initializer.TotalPrimitiveCount += Segment.NumPrimitives;
					}

					FRayTracingGeometryRHIRef RayTracingGeometryRHI = RHICmdList.CreateRayTracingGeometry(Initializer);

					if (ScheduledBuildsNumPrimitives < GNaniteRayTracingMaxBuiltPrimitivesPerFrame)
					{
						ScheduledBuildsNumPrimitives += RayTracingGeometryRHI->GetInitializer().TotalPrimitiveCount;

						Data.RayTracingGeometryRHI = MoveTemp(RayTracingGeometryRHI);

						if (Data.AuxiliaryDataOffset != INDEX_NONE)
						{
							AuxiliaryDataAllocator.Free(Data.AuxiliaryDataOffset, Data.AuxiliaryDataSize);
						}
						// allocate persistent auxiliary range
						Data.AuxiliaryDataSize = CalculateAuxiliaryDataSizeInUints(Initializer.TotalPrimitiveCount);
						Data.AuxiliaryDataOffset = AuxiliaryDataAllocator.Allocate(Data.AuxiliaryDataSize);

						for (auto& Primitive : Data.Primitives)
						{
							if (bUsingNaniteRayTracing)
							{
								Primitive->SetCachedRayTracingInstanceGeometryRHI(Data.RayTracingGeometryRHI, Data.NumSegments);
							}

							auto NaniteProxy = static_cast<Nanite::FSceneProxyBase*>(Primitive->Proxy);
							NaniteProxy->SetRayTracingDataOffset(Data.AuxiliaryDataOffset);

							Primitive->Scene->GPUScene.AddPrimitiveToUpdate(Primitive->GetPersistentIndex(), EPrimitiveDirtyState::ChangedOther);
						}

						ScheduledBuilds.Add(GeometryId);
					}
					else
					{
						FPendingBuild PendingBuild;
						PendingBuild.GeometryId = GeometryId;
						PendingBuild.RayTracingGeometryRHI = MoveTemp(RayTracingGeometryRHI);
						PendingBuilds.Add(MoveTemp(PendingBuild));

						INC_DWORD_STAT_BY(STAT_NaniteRayTracingPendingBuilds, 1);
					}
				}

				ReadbackData.EntryGeometryId.Empty();
				ReadbackData.MeshDataReadbackBuffer->Unlock();
			}
			else
			{
				break;
			}
		}

		INC_DWORD_STAT_BY(STAT_NaniteRayTracingScheduledBuildsNumPrimitives, ScheduledBuildsNumPrimitives - PrevScheduledBuildsNumPrimitives);
	}

	void FRayTracingManager::ResetPageInstallationState()
	{
		//UE_LOGF(LogRenderer, Warning, "FRayTracingManager::ResetPageInstallationState()\n");

		// reset page installation state (forces all installed pages to be processed again)

		for (auto& PageInfo : PendingUninstalledPages)
		{
			InstalledPages.Remove(PageInfo.GPUPageIndex);
		}

		PendingUninstalledPages.Reset();

		for (const auto& InstalledPage : InstalledPages)
		{
			FPageInfo PageInfo;
			PageInfo.GPUPageIndex = InstalledPage.Key;
			PageInfo.NumClusters = InstalledPage.Value.NumClusters;
			PageInfo.RuntimeResourceID = InstalledPage.Value.RuntimeResourceID;

			PendingInstalledPages.Add(PageInfo);
		}

		InstalledPages.Reset();

		CLASDataAllocator.Reset();
	}

	uint32 FRayTracingManager::GetRequestedCLASAllocationSize() const
	{
		int32 RequestedSize = CVarNaniteRayTracingCLASAllocationSize.GetValueOnRenderThread();
		check(CLASMaxSize > 0);
		uint32 Size = RequestedSize > 0 ? FMath::Max((uint32)RequestedSize, CLASMaxSize) : CLASMaxSize;
		Size = AlignArbitrary(Size, GRHIGlobals.RayTracing.ClusterAccelerationStructureAlignment);
		return Size;
	}

	void FRayTracingManager::ResizeCLASBufferIfNeeded(FRDGBuilder& GraphBuilder)
	{
		// D3D12 limits resources to 2048MB.
		const int32 CLASBufferSizeMB = FMath::Clamp(CVarNaniteRayTracingCLASBufferSizeMB.GetValueOnRenderThread(), 16, 2048);
		const int32 CLASBufferSize = CLASBufferSizeMB * 1024 * 1024;

		FRDGBufferDesc CLASBufferDesc;
		CLASBufferDesc.Usage = EBufferUsageFlags::AccelerationStructure;
		CLASBufferDesc.BytesPerElement = GRHIGlobals.RayTracing.ClusterAccelerationStructureAlignment;
		CLASBufferDesc.NumElements = FMath::DivideAndRoundUp<uint32>(CLASBufferSize, CLASBufferDesc.BytesPerElement);

		const bool bReallocated = AllocatePooledBuffer(CLASBufferDesc, CLASBuffer, TEXT("NaniteRayTracing.CLASBuffer"));

		int32 RequestedAllocationSize = GetRequestedCLASAllocationSize();

		if (bReallocated || RequestedAllocationSize != CLASPoolAllocationSize)
		{
			CLASPoolAllocationSize = RequestedAllocationSize;
			CLASPoolSize = FMath::DivideAndRoundDown<uint32>(CLASBufferSize, CLASPoolAllocationSize);

			// 1 uint for count + N for free list
			{
				FRDGBufferDesc BufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(int32), 1 + CLASPoolSize);
				BufferDesc.Usage = EBufferUsageFlags(BufferDesc.Usage | BUF_SourceCopy);
				AllocatePooledBuffer(BufferDesc, CLASAllocatorBuffer, TEXT("NaniteRayTracing.CLASAllocator"));
			}

			{
				//ue-todo: Refactor to use EShaderPlatform directly instead of converting from FeatureLevel
				FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GShaderPlatformForFeatureLevel[GetFeatureLevel()]);

				FRDGBufferRef CLASAllocatorBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASAllocatorBuffer);
				FRDGBufferUAVRef CLASAllocatorBufferUAV = GraphBuilder.CreateUAV(CLASAllocatorBufferRDG);

				FNaniteRayTracingResetAllocatorCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingResetAllocatorCS::FParameters>();
				PassParameters->RWAllocator = CLASAllocatorBufferUAV;
				PassParameters->PoolSize = CLASPoolSize;

				FNaniteRayTracingResetAllocatorCS::FPermutationDomain PermutationVector;

				auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingResetAllocatorCS>(PermutationVector);
				FIntVector GroupCount = FComputeShaderUtils::GetGroupCountWrapped(CLASPoolSize, FNaniteRayTracingResetAllocatorCS::ThreadGroupSize);

				FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::ResetCLASAllocator"), ComputeShader, PassParameters, GroupCount);

				ResetPageInstallationState();
			}
		}
	}

	void FRayTracingManager::ResizeBLASCacheIfNeeded(FRDGBuilder& GraphBuilder)
	{
		const bool bEnableBLASCache = GNaniteRayTracingBLASCache && GNaniteRayTracingUseReferenceInstances;
		if (bEnableBLASCache)
		{
			const int32 BLASCacheSizeMB = FMath::Clamp(GNaniteRayTracingBLASCacheSizeMB, 1, 2048);
			const uint32 DesiredBLASCacheSize = AlignArbitrary((uint32)(BLASCacheSizeMB * 1024 * 1024), GRHIGlobals.RayTracing.AccelerationStructureAlignment);
			if (!BLASCache || BLASCache->GetMaxSize() != DesiredBLASCacheSize)
			{
				BLASCache = MakeShared<FNaniteRayTracingBLASCache>(DesiredBLASCacheSize);
			}
		}
		else
		{
			BLASCache = nullptr;
		}
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FNaniteRayTracingResourceAddressesParams, )
		// TODO: Need way to specify buffer parameters so resources are created and address queried
		// Currently using UAVCompute to avoid asserts related to resources not yet produced...
		RDG_BUFFER_ACCESS(ResourceAddressesBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(VertexBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(IndexBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(GeometryIndexAndFlagsBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(CLASBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(BLASStagingBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(CLASAddressesBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(BLASCacheBuffer, ERHIAccess::UAVCompute)
	END_SHADER_PARAMETER_STRUCT()

	BEGIN_SHADER_PARAMETER_STRUCT(FNaniteRayTracingBuildCLASesParams, )
		RDG_BUFFER_ACCESS(ScratchBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(ClusterCountBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(DescriptorsBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(CLASSizesBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(CLASAddressesBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(CLASBuffer, ERHIAccess::BVHWrite)

		RDG_BUFFER_ACCESS(VertexBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(IndexBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(GeometryIndexAndFlagsBuffer, ERHIAccess::SRVCompute)
	END_SHADER_PARAMETER_STRUCT()

	BEGIN_SHADER_PARAMETER_STRUCT(FNaniteRayTracingMoveASesParams, )
		RDG_BUFFER_ACCESS(ScratchBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(CountBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(SrcAddressesBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(DstAddressesBuffer, ERHIAccess::SRVCompute)

		RDG_BUFFER_ACCESS(StagingBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(DstBuffer, ERHIAccess::BVHWrite)
	END_SHADER_PARAMETER_STRUCT()

	BEGIN_SHADER_PARAMETER_STRUCT(FNaniteRayTracingBuildBLASesParams, )
		RDG_BUFFER_ACCESS(ScratchBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(BLASCountBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(DescriptorsBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(BLASSizesBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(BLASAddressesBuffer, ERHIAccess::UAVCompute)
		RDG_BUFFER_ACCESS(BLASStagingBuffer, ERHIAccess::BVHWrite)

		RDG_BUFFER_ACCESS(CLASAddressesBuffer, ERHIAccess::SRVCompute)
		RDG_BUFFER_ACCESS(CLASBuffer, ERHIAccess::SRVCompute)
	END_SHADER_PARAMETER_STRUCT()

	FRayTracingManager::FClusterStreamOutBuffers::FClusterStreamOutBuffers(uint32 MaxClusters)
		: MaxClusters(MaxClusters)
		, MaxNumVertices(MaxClusters * NANITE_MAX_CLUSTER_VERTICES)
		, MaxNumIndices(MaxClusters * NANITE_MAX_CLUSTER_TRIANGLES * 3)
		, MaxNumTriangles(MaxClusters * NANITE_MAX_CLUSTER_TRIANGLES)
	{
		// Query sizes for CLAS build
		ClusterBuildOpInitializer.MaxResultCount = MaxClusters;
		ClusterBuildOpInitializer.Type = ERayTracingClusterOperationType::CLAS_BUILD;
		ClusterBuildOpInitializer.Mode = ERayTracingClusterOperationMode::IMPLICIT_DESTINATIONS;
		ClusterBuildOpInitializer.Flags = ERayTracingClusterOperationFlags::FAST_TRACE;
		ClusterBuildOpInitializer.Operation.CLAS.VertexFormat = VET_Float3;
		ClusterBuildOpInitializer.Operation.CLAS.MaxGeometryIndex = 1;
		ClusterBuildOpInitializer.Operation.CLAS.MaxUniqueGeometryCount = 1;
		ClusterBuildOpInitializer.Operation.CLAS.MaxTriangleCount = NANITE_MAX_CLUSTER_TRIANGLES;
		ClusterBuildOpInitializer.Operation.CLAS.MaxVertexCount = NANITE_MAX_CLUSTER_VERTICES;
		ClusterBuildOpInitializer.Operation.CLAS.MaxTotalTriangleCount = NANITE_MAX_CLUSTER_TRIANGLES * MaxClusters;
		ClusterBuildOpInitializer.Operation.CLAS.MaxTotalVertexCount = NANITE_MAX_CLUSTER_VERTICES * MaxClusters;

		ClusterBuildOpSize = RHICalcRayTracingClusterOperationSize(ClusterBuildOpInitializer);
	}

	void FRayTracingManager::FClusterStreamOutBuffers::Initialize(FRDGBuilder& GraphBuilder)
	{
		FRDGBufferDesc ClusterVertexBufferDesc = FRDGBufferDesc::CreateByteAddressDesc(sizeof(float) * 3 * FMath::Max(MaxNumVertices, 32U));
		ClusterVertexBuffer = GraphBuilder.CreateBuffer(ClusterVertexBufferDesc, TEXT("NaniteRayTracing.CLASVertexBuffer"));

		FRDGBufferDesc ClusterIndexBufferDesc = FRDGBufferDesc::CreateByteAddressDesc(sizeof(uint32) * FMath::Max(MaxNumIndices, 32U));
		ClusterIndexBuffer = GraphBuilder.CreateBuffer(ClusterIndexBufferDesc, TEXT("NaniteRayTracing.CLASIndexBuffer"));

		FRDGBufferDesc ClusterGeometryIndexAndFlagsBufferDesc = FRDGBufferDesc::CreateByteAddressDesc(sizeof(uint32) * FMath::Max(MaxNumTriangles, 32U));
		ClusterGeometryIndexAndFlagsBuffer = GraphBuilder.CreateBuffer(ClusterGeometryIndexAndFlagsBufferDesc, TEXT("NaniteRayTracing.CLASGeometryIndexAndFlagsBuffer"));

		FRDGBufferDesc ClusterCountBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 1);
		ClusterCountBuffer = GraphBuilder.CreateBuffer(ClusterCountBufferDesc, TEXT("NaniteRayTracing.CLASCountBuffer"));

		FRDGBufferDesc ClusterBuildDescBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(RAYTRACING_CLUSTER_OPS_BUILD_CLAS_DESC), MaxClusters);
		ClusterBuildDescBuffer = GraphBuilder.CreateBuffer(ClusterBuildDescBufferDesc, TEXT("NaniteRayTracing.CLASBuildDescBuffer"));

		// For CLAS build we have a fixed size so can ensure we have at least enough scratch/staging for that up front
		RequestScratch(GraphBuilder, ClusterBuildOpSize.ScratchSizeInBytes);
		RequestStaging(GraphBuilder, ClusterBuildOpSize.ResultMaxSizeInBytes);
	}

	void FRayTracingManager::FClusterStreamOutBuffers::RequestScratch(FRDGBuilder& GraphBuilder, uint32 Size)
	{
		if (Size > ScratchSize)
		{
			// TODO: Experiment with this heuristic
			ScratchSize = FMath::RoundUpToPowerOfTwo(Size);
			ScratchBuffer = nullptr;
		}

		if (!ScratchBuffer)
		{
			FRDGBufferDesc BufferDesc;
			BufferDesc.Usage = EBufferUsageFlags::RayTracingScratch | EBufferUsageFlags::StructuredBuffer;
			BufferDesc.BytesPerElement = GRHIGlobals.RayTracing.ScratchBufferAlignment;
			BufferDesc.NumElements = FMath::DivideAndRoundUp<uint32>(ScratchSize, BufferDesc.BytesPerElement);

			ScratchBuffer = GraphBuilder.CreateBuffer(BufferDesc, TEXT("NaniteRayTracing.ScratchBuffer"));
		}
	}

	void FRayTracingManager::FClusterStreamOutBuffers::RequestStaging(FRDGBuilder& GraphBuilder, uint32 Size)
	{
		if (Size > StagingSize)
		{
			// TODO: Experiment with this heuristic
			StagingSize = FMath::RoundUpToPowerOfTwo(Size);
			StagingBuffer = nullptr;
		}

		if (!StagingBuffer)
		{
			// This buffer may store CLASes or BLASes so take the most restrictive of the two alignments
			const uint32 Alignment = FMath::Max(GRHIGlobals.RayTracing.AccelerationStructureAlignment, GRHIGlobals.RayTracing.ClusterAccelerationStructureAlignment);

			FRDGBufferDesc BufferDesc;
			BufferDesc.Usage = EBufferUsageFlags::AccelerationStructure;
			BufferDesc.BytesPerElement = Alignment;
			BufferDesc.NumElements = FMath::DivideAndRoundUp<uint32>(StagingSize, BufferDesc.BytesPerElement);

			StagingBuffer = GraphBuilder.CreateBuffer(BufferDesc, TEXT("NaniteRayTracing.StagingBuffer"));
		}
	}

	void FRayTracingManager::FClusterStreamOutBuffers::Cleanup()
	{
		SET_MEMORY_STAT(STAT_NaniteRayTracingScratchSize, ScratchBuffer ? ScratchBuffer->GetSize() : 0U);
		SET_MEMORY_STAT(STAT_NaniteRayTracingStagingSize, StagingBuffer ? StagingBuffer->GetSize() : 0U);

		// Drop references to the RDG buffers
		ClusterVertexBuffer 				= nullptr;
		ClusterIndexBuffer 					= nullptr;
		ClusterGeometryIndexAndFlagsBuffer	= nullptr;
		ClusterCountBuffer 					= nullptr;
		ClusterBuildDescBuffer 				= nullptr;
		ScratchBuffer 						= nullptr;
		StagingBuffer 						= nullptr;
	}

	uint32 FRayTracingManager::GetMaxPageInstallBatchSize() const
	{
		return FMath::Max(1, CVarNaniteRayTracingMaxPageInstallBatchSize.GetValueOnRenderThread());
	}

	void FRayTracingManager::BuildCLASForPages(FRDGBuilder& GraphBuilder,
											   const TArray<FPageInfo>& Pages,
											   FRDGBufferRef ResourceAddressesBufferRDG)
	{
		check(StreamOutBuffers);
		check(StreamOutBuffers->MaxClusters >= (uint32)Pages.Num() * NANITE_MAX_CLUSTERS_PER_PAGE);

		//ue-todo: Refactor to use EShaderPlatform directly instead of converting from FeatureLevel
		FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GShaderPlatformForFeatureLevel[GetFeatureLevel()]);

		FRDGBufferRef CLASBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASBuffer);

		FRDGBufferRef CLASAllocatorBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASAllocatorBuffer);
		FRDGBufferUAVRef CLASAllocatorBufferUAV = GraphBuilder.CreateUAV(CLASAllocatorBufferRDG);

		uint32 NumInstalledClusters = 0;

		for (const FPageInfo& PageInfo : Pages)
		{
			// Invalidate any cached BLASes to allow a new cut to pick up newly streamed pages
			if (BLASCache)
			{
				BLASCache->InvalidateEntry(PageInfo.RuntimeResourceID);
			}

			NumInstalledClusters += PageInfo.NumClusters;
		}

		FRDGBufferRef CLASDataBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASDataBuffer);
		FRDGBufferRef PageDataBufferRDG = GraphBuilder.RegisterExternalBuffer(PageDataBuffer);
		FRDGBufferRef SegmentMappingBufferRDG = GraphBuilder.RegisterExternalBuffer(SegmentMappingBuffer);

		// Upload cluster stream out request buffer

		FRDGBufferRef StreamOutRequestBuffer = nullptr;
		FRDGBufferRef AllocateClustersRequestBuffer = nullptr;

		{
			FRDGUploadData<FClusterStreamOutRequest> StreamOutUploadData(GraphBuilder, NumInstalledClusters);
			FRDGUploadData<FAllocateClustersRequest> AllocateClustersUploadData(GraphBuilder, Pages.Num());

			uint32 StreamOutIndex = 0;
			uint32 AllocateClustersIndex = 0;

			uint32 NumInstalledClustersTmp = 0;

			for (const FPageInfo& PageInfo : Pages)
			{
				for (uint32 ClusterIndex = 0; ClusterIndex < PageInfo.NumClusters; ++ClusterIndex)
				{
					FClusterStreamOutRequest& Request = StreamOutUploadData[StreamOutIndex];
					Request.GPUPageIndex = PageInfo.GPUPageIndex;
					Request.ClusterIndex = ClusterIndex;
					Request.VertexBufferOffset = StreamOutIndex * NANITE_MAX_CLUSTER_VERTICES * 3 * sizeof(float);
					Request.IndexBufferOffset = StreamOutIndex * NANITE_MAX_CLUSTER_TRIANGLES * 3 * sizeof(uint32);
					Request.GeometryIndexAndFlagsBufferOffset = StreamOutIndex * NANITE_MAX_CLUSTER_TRIANGLES * sizeof(uint32);
					Request.SegmentMappingOffset = INDEX_NONE;

					const uint32* GeometryId = ResourceToRayTracingIdMap.Find(PageInfo.RuntimeResourceID);
					if (GeometryId != nullptr)
					{
						FInternalData& Data = *Geometries[*GeometryId];
						Request.SegmentMappingOffset = Data.SegmentMappingOffset;
					}

					++StreamOutIndex;
				}

				check(InstalledPages.Contains(PageInfo.GPUPageIndex));
				FInstalledPage& InstalledPage = *InstalledPages.Find(PageInfo.GPUPageIndex);

				FAllocateClustersRequest& Request = AllocateClustersUploadData[AllocateClustersIndex];
				Request.BaseCLASIndex = NumInstalledClustersTmp;
				Request.NumClusters = InstalledPage.NumClusters;
				Request.GPUPageIndex = PageInfo.GPUPageIndex;

				++AllocateClustersIndex;

				NumInstalledClustersTmp += PageInfo.NumClusters;
			}

			StreamOutRequestBuffer = CreateStructuredBuffer(GraphBuilder, TEXT("NaniteRayTracing.StreamOutRequestBuffer"), StreamOutUploadData);
			AllocateClustersRequestBuffer = CreateStructuredBuffer(GraphBuilder, TEXT("NaniteRayTracing.AllocateClustersRequestBuffer"), AllocateClustersUploadData);
		}

		StreamOutClusters(
			GraphBuilder,
			ShaderMap,
			NumInstalledClusters,
			StreamOutRequestBuffer,
			ResourceAddressesBufferRDG,
			SegmentMappingBufferRDG);

		FRDGBufferDesc CLASSizesBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), NumInstalledClusters);
		FRDGBufferRef CLASSizesBuffer = GraphBuilder.CreateBuffer(CLASSizesBufferDesc, TEXT("NaniteRayTracing.CLASSizesBuffer"));

		FRDGBufferDesc CLASAddressesBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(GPU_VIRTUAL_ADDRESS), NumInstalledClusters);
		FRDGBufferRef CLASAddressesBuffer = GraphBuilder.CreateBuffer(CLASAddressesBufferDesc, TEXT("NaniteRayTracing.CLASAddressesBuffer"));

		{
			FNaniteRayTracingBuildCLASesParams* PassParams = GraphBuilder.AllocParameters<FNaniteRayTracingBuildCLASesParams>();
			PassParams->ScratchBuffer = StreamOutBuffers->ScratchBuffer;
			PassParams->ClusterCountBuffer = StreamOutBuffers->ClusterCountBuffer;
			PassParams->DescriptorsBuffer = StreamOutBuffers->ClusterBuildDescBuffer;
			PassParams->CLASSizesBuffer = CLASSizesBuffer;
			PassParams->CLASAddressesBuffer = CLASAddressesBuffer;
			PassParams->CLASBuffer = StreamOutBuffers->StagingBuffer;
			PassParams->VertexBuffer = StreamOutBuffers->ClusterVertexBuffer;
			PassParams->IndexBuffer = StreamOutBuffers->ClusterIndexBuffer;
			PassParams->GeometryIndexAndFlagsBuffer = StreamOutBuffers->ClusterGeometryIndexAndFlagsBuffer;

			GraphBuilder.AddPass(RDG_EVENT_NAME("NaniteRayTracing::BuildCLASes"), PassParams, ERDGPassFlags::Compute,
				[PassParams, ClusterOpInitializer = StreamOutBuffers->ClusterBuildOpInitializer](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
				{
					FRayTracingClusterOperationParams OpParams = {};
					OpParams.Initializer = ClusterOpInitializer;

					OpParams.Resources.In.Scratch = PassParams->ScratchBuffer->GetRHI();
					OpParams.Resources.In.Descriptors = PassParams->DescriptorsBuffer->GetRHI();
					OpParams.Resources.In.ResultCount = PassParams->ClusterCountBuffer->GetRHI();

					OpParams.Resources.InOut.Addresses = PassParams->CLASAddressesBuffer->GetRHI();
					OpParams.Resources.Out.Sizes = PassParams->CLASSizesBuffer->GetRHI();
					OpParams.Resources.Out.AccelerationStructures = PassParams->CLASBuffer->GetRHI();

					OpParams.Resources.In.AdditionalResources.Add(PassParams->VertexBuffer->GetRHI());
					OpParams.Resources.In.AdditionalResources.Add(PassParams->IndexBuffer->GetRHI());
					OpParams.Resources.In.AdditionalResources.Add(PassParams->GeometryIndexAndFlagsBuffer->GetRHI());

					RHICmdList.ExecuteMultiIndirectClusterOperation(OpParams);
				});
		}

		FRDGBufferRef MoveCountBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 1), TEXT("NaniteRayTracing.CLAS.MoveCountBuffer"));
		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(MoveCountBuffer), 0);

		FRDGBufferDesc AddressesBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(GPU_VIRTUAL_ADDRESS), NumInstalledClusters);
		FRDGBufferRef SrcAddressesBuffer = GraphBuilder.CreateBuffer(AddressesBufferDesc, TEXT("NaniteRayTracing.CLAS.SrcAddressesBuffer"));
		FRDGBufferRef DstAddressesBuffer = GraphBuilder.CreateBuffer(AddressesBufferDesc, TEXT("NaniteRayTracing.CLAS.DstAddressesBuffer"));

		{
			FNaniteRayTracingAllocateClusterCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingAllocateClusterCS::FParameters>();
			PassParameters->AllocateClustersRequests = GraphBuilder.CreateSRV(AllocateClustersRequestBuffer);
			PassParameters->NumAllocateClustersRequests = Pages.Num();
			PassParameters->CLASSizes = GraphBuilder.CreateSRV(CLASSizesBuffer);
			PassParameters->CLASAddresses = GraphBuilder.CreateSRV(CLASAddressesBuffer);
			PassParameters->NumCLAS = NumInstalledClusters;
			PassParameters->ResourceAddresses = GraphBuilder.CreateSRV(ResourceAddressesBufferRDG);
			PassParameters->RWAllocator = CLASAllocatorBufferUAV;
			PassParameters->AllocationSize = CLASPoolAllocationSize;
			PassParameters->PoolSize = CLASPoolSize;
			PassParameters->CLASBufferSize = CLASBuffer->GetSize();
			PassParameters->CLASAlignment = GRHIGlobals.RayTracing.ClusterAccelerationStructureAlignment;
			PassParameters->RWMoveCount = GraphBuilder.CreateUAV(MoveCountBuffer);
			PassParameters->RWSrcAddresses = GraphBuilder.CreateUAV(SrcAddressesBuffer);
			PassParameters->RWDstAddresses = GraphBuilder.CreateUAV(DstAddressesBuffer);
			PassParameters->RWPageDataBuffer = GraphBuilder.CreateUAV(PageDataBufferRDG);
			PassParameters->RWCLASData = GraphBuilder.CreateUAV(CLASDataBufferRDG);

			FNaniteRayTracingAllocateClusterCS::FPermutationDomain PermutationVector;

			auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingAllocateClusterCS>(PermutationVector);
			FIntVector GroupCount = FComputeShaderUtils::GetGroupCountWrapped(NumInstalledClusters, FNaniteRayTracingAllocateClusterCS::ThreadGroupSize);

			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::AllocateClusters"), ComputeShader, PassParameters, GroupCount);
		}

		{
			FRayTracingClusterOperationInitializer MoveOpInitializer = {};
			MoveOpInitializer.MaxResultCount = NumInstalledClusters;
			MoveOpInitializer.Type = ERayTracingClusterOperationType::MOVE;
			MoveOpInitializer.Mode = ERayTracingClusterOperationMode::EXPLICIT_DESTINATIONS;
			MoveOpInitializer.Flags = ERayTracingClusterOperationFlags::NONE;
			MoveOpInitializer.Operation.Move.Type = ERayTracingClusterOperationMoveType::CLUSTER_LEVEL;
			MoveOpInitializer.Operation.Move.MaxBytes = StreamOutBuffers->ClusterBuildOpSize.ResultMaxSizeInBytes;

			FRayTracingClusterOperationSize MoveOpSize = RHICalcRayTracingClusterOperationSize(MoveOpInitializer);
			StreamOutBuffers->RequestScratch(GraphBuilder, MoveOpSize.ScratchSizeInBytes);

			FNaniteRayTracingMoveASesParams* PassParams = GraphBuilder.AllocParameters<FNaniteRayTracingMoveASesParams>();
			PassParams->ScratchBuffer = StreamOutBuffers->ScratchBuffer;
			PassParams->CountBuffer = MoveCountBuffer;
			PassParams->SrcAddressesBuffer = SrcAddressesBuffer;
			PassParams->DstAddressesBuffer = DstAddressesBuffer;
			PassParams->StagingBuffer = StreamOutBuffers->StagingBuffer;
			PassParams->DstBuffer = CLASBufferRDG;

			GraphBuilder.AddPass(RDG_EVENT_NAME("NaniteRayTracing::MoveCLASes"), PassParams, ERDGPassFlags::Compute,
				[PassParams, MoveOpInitializer](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
				{
					FRayTracingClusterOperationParams OpParams = {};
					OpParams.Initializer = MoveOpInitializer;

					OpParams.Resources.In.Scratch = PassParams->ScratchBuffer->GetRHI();
					OpParams.Resources.In.Descriptors = PassParams->SrcAddressesBuffer->GetRHI();
					OpParams.Resources.In.ResultCount = PassParams->CountBuffer->GetRHI();

					OpParams.Resources.InOut.Addresses = PassParams->DstAddressesBuffer->GetRHI();
					//OpParams.Resources.Out.Sizes = PassParams->CLASSizesBuffer->GetRHI();

					OpParams.Resources.In.AdditionalResources.Add(PassParams->StagingBuffer->GetRHI());
					OpParams.Resources.In.AdditionalResources.Add(PassParams->DstBuffer->GetRHI());

					RHICmdList.ExecuteMultiIndirectClusterOperation(OpParams);
				});
		}

		// TODO: This can maybe move to ProcessPendingInstalledPages? Not sure if it needs to happen every iteration
		{
			FNaniteRayTracingClampAllocatorCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingClampAllocatorCS::FParameters>();
			PassParameters->RWAllocator = CLASAllocatorBufferUAV;
			PassParameters->PoolSize = CLASPoolSize;

			FNaniteRayTracingClampAllocatorCS::FPermutationDomain PermutationVector;

			auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingClampAllocatorCS>(PermutationVector);
			FIntVector GroupCount = FIntVector(1, 1, 1);

			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::ClampCLASAllocator"), ComputeShader, PassParameters, GroupCount);
		}

		INC_DWORD_STAT_BY(STAT_NaniteRayTracingCLASBuilds, NumInstalledClusters);
	}

	void FRayTracingManager::ProcessPendingInstalledPages(FRDGBuilder& GraphBuilder)
	{
		if (PendingInstalledPages.IsEmpty())
		{
			return;
		}
		
		uint32 NumInstalledClusters = 0;

		for (const FPageInfo& PageInfo : PendingInstalledPages)
		{
			NumInstalledClusters += PageInfo.NumClusters;
		}

		FNaniteRayTracingASData NullCLASData;
		NullCLASData.Address = 0;
		NullCLASData.Size = 0;
		NullCLASData.AllocationIndex = uint32(-1);

		PendingCLASBuildPages.Reserve(PendingCLASBuildPages.Num() + PendingInstalledPages.Num());

		PageDataUploadBuffer.Init(GraphBuilder, PendingInstalledPages.Num(), sizeof(FNaniteRayTracingPageData), false, TEXT("NaniteRayTracing.PageDataUploadBuffer"));
		CLASDataUploadBuffer.Init(GraphBuilder, NumInstalledClusters, sizeof(FNaniteRayTracingASData), false, TEXT("NaniteRayTracing.CLASDataUploadBuffer"));

		for (const FPageInfo& PageInfo : PendingInstalledPages)
		{
			check(!InstalledPages.Contains(PageInfo.GPUPageIndex));

			FInstalledPage& InstalledPage = InstalledPages.FindOrAdd(PageInfo.GPUPageIndex, {});
			InstalledPage.BaseIndex = CLASDataAllocator.Allocate(PageInfo.NumClusters);
			InstalledPage.NumClusters = PageInfo.NumClusters;
			InstalledPage.RuntimeResourceID = PageInfo.RuntimeResourceID;

			FNaniteRayTracingPageData PageData;
			PageData.BaseCLASDataIndex = InstalledPage.BaseIndex;
			PageData.NumClusters = InstalledPage.NumClusters;
			PageData.NumAllocatedBlocks = 0;

			PageDataUploadBuffer.Add(PageInfo.GPUPageIndex, &PageData);

			MaxGPUPageIndex = FMath::Max(MaxGPUPageIndex, PageInfo.GPUPageIndex);

			PendingCLASBuildPages.Add(PageInfo);

			for (uint32 Index = 0; Index < PageInfo.NumClusters; ++Index)
			{
				CLASDataUploadBuffer.Add(InstalledPage.BaseIndex + Index, &NullCLASData);
			}
		}

		// TODO: Replace RoundUpToPowerOfTwo
		FRDGBufferRef PageDataBufferRDG = ResizeByteAddressBufferIfNeeded(GraphBuilder, PageDataBuffer, sizeof(FNaniteRayTracingPageData) * FMath::RoundUpToPowerOfTwo(MaxGPUPageIndex + 1), TEXT("NaniteRayTracing.PageDataBuffer"), /*bCopy*/ true, EAllowShrinking::No);
		
		PageDataUploadBuffer.ResourceUploadTo(GraphBuilder, PageDataBufferRDG);

		// Ensure CLASDataBuffer is large enough for all new allocations
		FRDGBufferRef CLASDataBufferRDG = ResizeByteAddressBufferIfNeeded(GraphBuilder, CLASDataBuffer, sizeof(FNaniteRayTracingASData) * CLASDataAllocator.GetMaxSize(), TEXT("NaniteRayTracing.CLASDataBuffer"), /*bCopy*/ true, EAllowShrinking::No);

		CLASDataUploadBuffer.ResourceUploadTo(GraphBuilder, CLASDataBufferRDG);

		PendingInstalledPages.Empty();
	}

	void FRayTracingManager::ProcessPendingUninstalledPages(FRDGBuilder& GraphBuilder)
	{
		if (PendingUninstalledPages.IsEmpty())
		{
			return;
		}

		//ue-todo: Refactor to use EShaderPlatform directly instead of converting from FeatureLevel
		FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GShaderPlatformForFeatureLevel[GetFeatureLevel()]);

		FRDGBufferRef PageDataBufferRDG = GraphBuilder.RegisterExternalBuffer(PageDataBuffer);
		FRDGBufferRef CLASDataBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASDataBuffer);
		FRDGBufferRef CLASBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASBuffer);

		FRDGBufferRef CLASAllocatorBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASAllocatorBuffer);
		FRDGBufferUAVRef CLASAllocatorBufferUAV = GraphBuilder.CreateUAV(CLASAllocatorBufferRDG);

		// Upload page uninstall request buffer

		FRDGBufferRef RequestBuffer = nullptr;
		uint32 NumUninstalledClusters = 0;

		{
			FRDGUploadData<FFreeClustersRequest> UploadData(GraphBuilder, PendingUninstalledPages.Num());

			uint32 Index = 0;

			for (const FPageInfo& PageInfo : PendingUninstalledPages)
			{
				check(InstalledPages.Contains(PageInfo.GPUPageIndex));

				FInstalledPage& InstalledPage = *InstalledPages.Find(PageInfo.GPUPageIndex);
				check(InstalledPage.NumClusters == PageInfo.NumClusters);

				FFreeClustersRequest& Request = UploadData[Index];
				Request.GPUPageIndex = PageInfo.GPUPageIndex;

				NumUninstalledClusters += PageInfo.NumClusters;

				// Invalidate any cached BLASes associated with these pages
				if (BLASCache)
				{
					BLASCache->InvalidateEntry(PageInfo.RuntimeResourceID);
				}

				++Index;
			}

			RequestBuffer = CreateStructuredBuffer(GraphBuilder, TEXT("NaniteRayTracing.UninstallRequestBuffer"), UploadData);
		}

		{
			FNaniteRayTracingFreeClusterCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingFreeClusterCS::FParameters>();
			PassParameters->FreeClustersRequests = GraphBuilder.CreateSRV(RequestBuffer);
			PassParameters->NumFreeClustersRequests = PendingUninstalledPages.Num();
			PassParameters->NumCLAS = NumUninstalledClusters;
			PassParameters->RWAllocator = CLASAllocatorBufferUAV;
			PassParameters->AllocationSize = CLASPoolAllocationSize;
			PassParameters->PoolSize = CLASPoolSize;
			PassParameters->CLASBufferSize = CLASBufferRDG->GetSize();
			PassParameters->CLASAlignment = GRHIGlobals.RayTracing.ClusterAccelerationStructureAlignment;
			PassParameters->RWPageDataBuffer = GraphBuilder.CreateUAV(PageDataBufferRDG);
			PassParameters->RWCLASData = GraphBuilder.CreateUAV(CLASDataBufferRDG);

			FNaniteRayTracingFreeClusterCS::FPermutationDomain PermutationVector;

			auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingFreeClusterCS>(PermutationVector);
			FIntVector GroupCount = FComputeShaderUtils::GetGroupCountWrapped(PassParameters->NumFreeClustersRequests, FNaniteRayTracingFreeClusterCS::ThreadGroupSize);

			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::FreeClusters"), ComputeShader, PassParameters, GroupCount);
		}

		PageDataUploadBuffer.Init(GraphBuilder, PendingUninstalledPages.Num(), sizeof(FNaniteRayTracingPageData), false, TEXT("NaniteRayTracing.PageDataUploadBuffer"));

		for (FPageInfo& PageInfo : PendingUninstalledPages)
		{
			check(InstalledPages.Contains(PageInfo.GPUPageIndex));

			FInstalledPage& InstalledPage = *InstalledPages.Find(PageInfo.GPUPageIndex);

			CLASDataAllocator.Free(InstalledPage.BaseIndex, InstalledPage.NumClusters);

			InstalledPage.BaseIndex = UINT32_MAX;
			InstalledPage.NumClusters = 0;

			FNaniteRayTracingPageData PageData;
			PageData.BaseCLASDataIndex = InstalledPage.BaseIndex;
			PageData.NumClusters = InstalledPage.NumClusters;
			PageData.NumAllocatedBlocks = 0;

			PageDataUploadBuffer.Add(PageInfo.GPUPageIndex, &PageData);

			InstalledPages.Remove(PageInfo.GPUPageIndex);
		}

		PageDataUploadBuffer.ResourceUploadTo(GraphBuilder, PageDataBufferRDG);

		PendingUninstalledPages.Reset();
	}

	void FRayTracingManager::ProcessPendingCLASBuildPages(FRDGBuilder& GraphBuilder)
	{
		if (PendingCLASBuildPages.IsEmpty())
		{
			return;
		}

		FRDGBufferRef CLASBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASBuffer);
		FRDGBufferRef CLASAllocatorBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASAllocatorBuffer);

		// Update our addresses buffer
		// NOTE: This is only allowed to happen one time per frame/GraphBuilder due to constraints on
		// how RDG and the RHI handle the WriteResourceGPUAddresses pass below!
		FRDGBufferRef CLASResourceAddressesBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASResourceAddressesBuffer);
		{
			FNaniteRayTracingResourceAddressesParams* PassParams = GraphBuilder.AllocParameters<FNaniteRayTracingResourceAddressesParams>();
			PassParams->ResourceAddressesBuffer = CLASResourceAddressesBufferRDG;
			PassParams->VertexBuffer = StreamOutBuffers->ClusterVertexBuffer;
			PassParams->IndexBuffer = StreamOutBuffers->ClusterIndexBuffer;
			PassParams->GeometryIndexAndFlagsBuffer = StreamOutBuffers->ClusterGeometryIndexAndFlagsBuffer;
			PassParams->CLASBuffer = CLASBufferRDG;

			GraphBuilder.AddPass(
				RDG_EVENT_NAME("NaniteRayTracing::WriteCLASResourceGPUAddresses"),
				PassParams,
				ERDGPassFlags::Compute,
				[PassParams](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
				{
					FRHIResource* Tmp[4] =
					{
						PassParams->VertexBuffer->GetRHI(),
						PassParams->IndexBuffer->GetRHI(),
						PassParams->GeometryIndexAndFlagsBuffer->GetRHI(),
						PassParams->CLASBuffer->GetRHI()
					};
					TConstArrayView<FRHIResource*> ReferencedResources = MakeConstArrayView(Tmp, sizeof(Tmp) / sizeof(FRHIResource*));
					RHICmdList.WriteResourceGPUAddresses(ReferencedResources, PassParams->ResourceAddressesBuffer->GetRHI());
				});
		}

		{
			const uint32 BatchSize = GetMaxPageInstallBatchSize();

			TArray<FPageInfo> PagesToBuild;
			PagesToBuild.Reserve(BatchSize);

			for (auto It = PendingCLASBuildPages.CreateIterator(); It; ++It)
			{
				if (ResourceToRayTracingIdMap.Contains(It->RuntimeResourceID))
				{
					PagesToBuild.Add(*It);
					It.RemoveCurrent();
				}

				if ((uint32)PagesToBuild.Num() >= BatchSize)
				{
					BuildCLASForPages(GraphBuilder, PagesToBuild, CLASResourceAddressesBufferRDG);
					PagesToBuild.Reset();
				}
			}

			if (!PagesToBuild.IsEmpty())
			{
				BuildCLASForPages(GraphBuilder, PagesToBuild, CLASResourceAddressesBufferRDG);
				PagesToBuild.Reset();
			}
		}

		if (CLASReadbackBuffersNumPending < MaxReadbackBuffers)
		{
			FCLASReadbackData& CLASReadbackData = CLASReadbackDatas[CLASReadbackBuffersWriteIndex];

			AddReadbackBufferPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::Readback"), CLASAllocatorBufferRDG,
				[ReadbackBuffer = CLASReadbackData.ReadbackBuffer, CLASAllocatorBufferRDG](FRDGAsyncTask, FRHICommandList& RHICmdList)
				{
					ReadbackBuffer->EnqueueCopy(RHICmdList, CLASAllocatorBufferRDG->GetRHI(), 0u);
				});

			CLASReadbackData.PoolSize = CLASPoolSize;
			CLASReadbackData.AllocationSize = CLASPoolAllocationSize;

			CLASReadbackBuffersWriteIndex = (CLASReadbackBuffersWriteIndex + 1u) % MaxReadbackBuffers;
			CLASReadbackBuffersNumPending = FMath::Min(CLASReadbackBuffersNumPending + 1u, MaxReadbackBuffers);
		}

		SET_MEMORY_STAT(STAT_NaniteRayTracingCLASBufferSize, CLASBuffer->GetSize());
	}

	void FRayTracingManager::ProcessPendingPages(FRDGBuilder& GraphBuilder)
	{
		RDG_EVENT_SCOPE(GraphBuilder, "NaniteRayTracing::ProcessPendingPages");

		//ue-todo: Refactor to use EShaderPlatform directly instead of converting from FeatureLevel
		FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GShaderPlatformForFeatureLevel[GetFeatureLevel()]);

		ProcessPendingUninstalledPages(GraphBuilder);

		ProcessPendingInstalledPages(GraphBuilder);

		ProcessPendingCLASBuildPages(GraphBuilder);

		while (CLASReadbackBuffersNumPending > 0)
		{
			uint32 Index = (CLASReadbackBuffersWriteIndex + MaxReadbackBuffers - CLASReadbackBuffersNumPending) % MaxReadbackBuffers;
			FCLASReadbackData& CLASReadbackData = CLASReadbackDatas[Index];
			if (CLASReadbackData.ReadbackBuffer->IsReady())
			{
				CLASReadbackBuffersNumPending--;

				auto ReadbackBufferPtr = (const uint32*)CLASReadbackData.ReadbackBuffer->Lock(1 * sizeof(uint32));

				const uint32 NumFreeAllocations = ReadbackBufferPtr[0];
				CLASBufferAllocatedSize = (CLASReadbackData.PoolSize - NumFreeAllocations) * CLASReadbackData.AllocationSize;

				bCLASBufferFull = (NumFreeAllocations == 0);

				CLASReadbackData.ReadbackBuffer->Unlock();
			}
			else
			{
				break;
			}
		}

		SET_MEMORY_STAT(STAT_NaniteRayTracingCLASAllocatedSize, CLASBufferAllocatedSize);
	}

	void FRayTracingManager::UpdateBLAS(FRDGBuilder& GraphBuilder)
	{
		//if (UpdateRequests.IsEmpty())
		if (Geometries.IsEmpty())
		{
			return;
		}

		RDG_EVENT_SCOPE(GraphBuilder, "NaniteRayTracing::UpdateBLAS");

		//ue-todo: Refactor to use EShaderPlatform directly instead of converting from FeatureLevel
		FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GShaderPlatformForFeatureLevel[GetFeatureLevel()]);

		uint32 MaxCLASPerBLASCount = 0;
		uint32 MaxTotalCLASCount = 0;

		//for (uint32 GeometryId : UpdateRequests)
		//{
		//	FInternalData& Data = *Geometries[GeometryId];
		for (FInternalData* Data : Geometries)
		{
			check(Data->NumResidentClustersUpdate > 0);

			MaxCLASPerBLASCount = FMath::Max(MaxCLASPerBLASCount, Data->NumResidentClustersUpdate);
			MaxTotalCLASCount += Data->NumResidentClustersUpdate;
		}

		FRDGBufferDesc CLASAddressesBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(GPU_VIRTUAL_ADDRESS), MaxTotalCLASCount);
		FRDGBufferRef CLASAddressesBuffer = GraphBuilder.CreateBuffer(CLASAddressesBufferDesc, TEXT("NaniteRayTracing.CLASAddressesBuffer"));

		FRDGBufferDesc BLASBuildBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(RAYTRACING_CLUSTER_OPS_BUILD_BLAS_DESC), Geometries.Num());
		FRDGBufferRef BLASBuildBuffer = GraphBuilder.CreateBuffer(BLASBuildBufferDesc, TEXT("NaniteRayTracing.BLASBuildBuffer"));

		FRDGBufferDesc BLASCountBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 1);
		FRDGBufferRef BLASCountBuffer = GraphBuilder.CreateBuffer(BLASCountBufferDesc, TEXT("NaniteRayTracing.BLASCountBuffer"));

		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(BLASCountBuffer), 0u);

		FRDGBufferRef ReferenceErrorsBufferRDG = ReferenceErrorsBuffer ? GraphBuilder.RegisterExternalBuffer(ReferenceErrorsBuffer) : GSystemTextures.GetDefaultStructuredBuffer(GraphBuilder, sizeof(uint32));

		FRDGBufferRef CLASBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASBuffer);

		FRDGBufferRef PageDataBufferRDG = GraphBuilder.RegisterExternalBuffer(PageDataBuffer);
		FRDGBufferRef CLASDataBufferRDG = GraphBuilder.RegisterExternalBuffer(CLASDataBuffer);
				
		FRDGBufferRef RequestBuffer = nullptr;
		uint32 NumRequests = 0;

		{
			FRDGUploadData<FGeometryClusterCutRequest> UploadData(GraphBuilder, Geometries.Num());

			uint32 BaseAddressIndex = 0;

			for (FInternalData* Data : Geometries)
			{
				FGeometryClusterCutRequest& Request = UploadData[NumRequests];
				Request.HierarchyOffset = Data->HierarchyOffset;
				Request.BaseAddressIndex = BaseAddressIndex;
				Request.RuntimeResourceID = Data->RuntimeResourceID;

				BaseAddressIndex += Data->NumResidentClustersUpdate;
				++NumRequests;
			}

			RequestBuffer = CreateStructuredBuffer(GraphBuilder, TEXT("NaniteRayTracing.GeometryClusterCutRequestBuffer"), UploadData);
		}

		// Create and clear BLASDataBuffer before InitQueueCS so it can write per-geometry build state
		FRDGBufferRef BLASDataBufferRDG = ResizeByteAddressBufferIfNeeded(GraphBuilder, BLASDataBuffer, sizeof(FNaniteRayTracingASData) * (MaxRootPageIndex + 1), TEXT("NaniteRayTracing.BLASDataBuffer"), /*bCopy*/ true, EAllowShrinking::No);
		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(BLASDataBufferRDG), 0);

		// Cache buffer setup with dummies for when the cache is disabled
		const bool bCacheEnabled = BLASCache != nullptr;
		FRDGBufferRef BLASCacheRequestBuffer;
		FRDGBufferSRV* BLASCacheAllocationTableSRV;
		FRDGBufferRef BLASCacheMetadataBufferRDG;
		FRDGBufferRef BLASCacheBufferRDG;
		const uint32 NumRootPages = Nanite::GStreamingManager.GetMaxRootPages();
		if (bCacheEnabled)
		{
			BLASCacheRequestBuffer = BLASCache->CreateRequestBuffer(GraphBuilder, NumRootPages);
			BLASCacheAllocationTableSRV = BLASCache->UploadAllocationTable(GraphBuilder);
			BLASCacheMetadataBufferRDG = BLASCache->RegisterOrCreateMetadataBuffer(GraphBuilder);
			BLASCacheBufferRDG = GraphBuilder.RegisterExternalBuffer(BLASCache->GetBuffer());
		}
		else
		{
			BLASCacheRequestBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateByteAddressDesc(sizeof(FNaniteRayTracingASCacheRequest)), TEXT("NaniteRayTracing.BLAS.CacheRequest.Dummy"));
			BLASCacheAllocationTableSRV = GraphBuilder.CreateSRV(GSystemTextures.GetDefaultStructuredBuffer(GraphBuilder, sizeof(FNaniteRayTracingASCacheEntry)));
			BLASCacheMetadataBufferRDG = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(FNaniteRayTracingASCacheMetadata), 1), TEXT("NaniteRayTracing.BLASCache.Metadata.Dummy"));
			// Use CLASBufferRDG as a dummy; it will not be referenced in the cache disabled path
			BLASCacheBufferRDG = CLASBufferRDG;
		}

		FRDGBufferRef ResourceAddressesBufferRDG = GraphBuilder.RegisterExternalBuffer(BLASResourceAddressesBuffer);

		FRayTracingClusterOperationInitializer BLASOpInitializer = {};
		BLASOpInitializer.MaxResultCount = Geometries.Num();
		BLASOpInitializer.Type = ERayTracingClusterOperationType::BLAS_BUILD;
		BLASOpInitializer.Mode = ERayTracingClusterOperationMode::IMPLICIT_DESTINATIONS;
		BLASOpInitializer.Flags = ERayTracingClusterOperationFlags::FAST_TRACE;
		BLASOpInitializer.Operation.BLAS.MaxCLASPerBLASCount = MaxCLASPerBLASCount;
		BLASOpInitializer.Operation.BLAS.MaxTotalCLASCount = MaxTotalCLASCount;

		FRayTracingClusterOperationSize BLASOpSize = RHICalcRayTracingClusterOperationSize(BLASOpInitializer);

		check(StreamOutBuffers);
		StreamOutBuffers->RequestScratch(GraphBuilder, BLASOpSize.ScratchSizeInBytes);

		// Grow the persistent BLAS staging buffer to fit the build output (round up to power of two, never shrink)
		{
			const uint32 RequiredSize = BLASOpSize.ResultMaxSizeInBytes;
			if (!BLASStagingBuffer || BLASStagingBuffer->GetSize() < RequiredSize)
			{
				const uint32 NewSize = FMath::RoundUpToPowerOfTwo(RequiredSize);
				FRDGBufferDesc Desc;
				Desc.Usage = EBufferUsageFlags::AccelerationStructure;
				Desc.BytesPerElement = GRHIGlobals.RayTracing.AccelerationStructureAlignment;
				Desc.NumElements = FMath::DivideAndRoundUp<uint32>(NewSize, Desc.BytesPerElement);
				AllocatePooledBuffer(Desc, BLASStagingBuffer, TEXT("NaniteRayTracing.BLASStagingBuffer"));
			}
		}
		FRDGBufferRef BLASStagingBufferRDG = GraphBuilder.RegisterExternalBuffer(BLASStagingBuffer);

		{
			FNaniteRayTracingResourceAddressesParams* PassParams = GraphBuilder.AllocParameters<FNaniteRayTracingResourceAddressesParams>();
			PassParams->ResourceAddressesBuffer = ResourceAddressesBufferRDG;
			PassParams->BLASStagingBuffer = BLASStagingBufferRDG;
			PassParams->CLASAddressesBuffer = CLASAddressesBuffer;
			PassParams->BLASCacheBuffer = BLASCacheBufferRDG;

			GraphBuilder.AddPass(
				RDG_EVENT_NAME("NaniteRayTracing::WriteResourceGPUAddresses_BLAS"),
				PassParams,
				ERDGPassFlags::Compute,
				[PassParams](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
				{
					FRHIResource* Tmp[3] = { PassParams->BLASStagingBuffer->GetRHI(), PassParams->CLASAddressesBuffer->GetRHI(), PassParams->BLASCacheBuffer->GetRHI() };
					TConstArrayView<FRHIResource*> ReferencedResources = MakeConstArrayView(Tmp, sizeof(Tmp) / sizeof(FRHIResource*));
					RHICmdList.WriteResourceGPUAddresses(ReferencedResources, PassParams->ResourceAddressesBuffer->GetRHI());
				});
		}

		FRayTracingQueueParameters QueueParameters;
		GetQueueParams(GraphBuilder, ShaderMap, /*bStreamingRequestsOnly*/ false, QueueParameters);

		// Reset queue to empty state
		AddClearUAVPass(GraphBuilder, QueueParameters.QueueState, 0);

		// Init queue with requests
		{
			FRayTracingGeometryClusterCutInitQueueCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FRayTracingGeometryClusterCutInitQueueCS::FParameters>();
			PassParameters->QueueParameters = QueueParameters;

			PassParameters->CutRequests = GraphBuilder.CreateSRV(RequestBuffer);
			PassParameters->NumCutRequests = NumRequests;

			PassParameters->ResourceAddressesBuffer = GraphBuilder.CreateSRV(ResourceAddressesBufferRDG);
			PassParameters->RWBLASBuildBuffer = GraphBuilder.CreateUAV(BLASBuildBuffer);

			PassParameters->BLASCacheAllocationTable = BLASCacheAllocationTableSRV;
			PassParameters->RWBLASCacheMetadata = GraphBuilder.CreateUAV(BLASCacheMetadataBufferRDG);
			PassParameters->RWBLASData = GraphBuilder.CreateUAV(BLASDataBufferRDG);
			PassParameters->RWBLASCountBuffer = GraphBuilder.CreateUAV(BLASCountBuffer);

			PassParameters->ReferenceErrors = GraphBuilder.CreateSRV(ReferenceErrorsBufferRDG);
			PassParameters->BLASCacheEnabled = bCacheEnabled ? 1 : 0;
			PassParameters->CacheRelativeErrorTolerance = GNaniteRayTracingBLASCacheRelativeErrorTolerance;

			ShaderPrint::SetParameters(GraphBuilder, PassParameters->ShaderPrint);

			FRayTracingGeometryClusterCutInitQueueCS::FPermutationDomain PermutationVector;
			SetStatsArgsAndPermutation<FRayTracingGeometryClusterCutInitQueueCS>(StatsBufferUAV, PassParameters, PermutationVector);

			auto ComputeShader = ShaderMap->GetShader<FRayTracingGeometryClusterCutInitQueueCS>(PermutationVector);

			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::InitQueue"), ComputeShader, PassParameters, FComputeShaderUtils::GetGroupCountWrapped(NumRequests, FRayTracingGeometryClusterCutInitQueueCS::ThreadGroupSize));
		}

		{
			RDG_EVENT_SCOPE(GraphBuilder, "HierarchyTraversal");

			FNaniteRayTracingGeometryClusterCutTraversalCS::FParameters SharedParameters;

			SharedParameters.QueueParameters = QueueParameters;

			SharedParameters.HierarchyBuffer = Nanite::GStreamingManager.GetHierarchySRV(GraphBuilder);
			SharedParameters.ClusterPageData = Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
			SharedParameters.PageConstants.X = 0;
			SharedParameters.PageConstants.Y = Nanite::GStreamingManager.GetMaxStreamingPages();

			SharedParameters.CutRequests = GraphBuilder.CreateSRV(RequestBuffer);
			SharedParameters.NumCutRequests = NumRequests;
			
			SharedParameters.ReferenceErrors = GraphBuilder.CreateSRV(ReferenceErrorsBufferRDG);
			SharedParameters.UseReferenceErrors = ReferenceErrorsBuffer ? 1 : 0;

			SharedParameters.MinCutError = GNaniteRayTracingMinCutError;

			ShaderPrint::SetParameters(GraphBuilder, SharedParameters.ShaderPrint);

			// Node passes
			{
				FRDGBufferRef NodeCullArgs0 = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc((NANITE_MAX_CLUSTER_HIERARCHY_DEPTH + 1) * NANITE_NODE_CULLING_ARG_COUNT), TEXT("Nanite.CullArgs0"));
				FRDGBufferRef NodeCullArgs1 = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc((NANITE_MAX_CLUSTER_HIERARCHY_DEPTH + 1) * NANITE_NODE_CULLING_ARG_COUNT), TEXT("Nanite.CullArgs1"));

				AddPass_InitNodeCullArgs(GraphBuilder, ShaderMap, RDG_EVENT_NAME("InitNodeCullArgs"), QueueParameters.QueueState, NodeCullArgs0, NodeCullArgs1, 0);

				FNaniteRayTracingGeometryClusterCutTraversalCS::FPermutationDomain PermutationVector;
				PermutationVector.Set<FNaniteRayTracingGeometryClusterCutTraversalCS::FCullingTypeDim>(NANITE_CULLING_TYPE_NODES);
				auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingGeometryClusterCutTraversalCS>(PermutationVector);

				const uint32 MaxLevels = Nanite::GStreamingManager.GetMaxHierarchyLevels();
				for (uint32 NodeLevel = 0; NodeLevel < MaxLevels; NodeLevel++)
				{
					auto* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingGeometryClusterCutTraversalCS::FParameters>(&SharedParameters);

					FRDGBufferRef CurrentIndirectArgs = (NodeLevel & 1) ? NodeCullArgs1 : NodeCullArgs0;
					FRDGBufferRef NextIndirectArgs = (NodeLevel & 1) ? NodeCullArgs0 : NodeCullArgs1;

					PassParameters->CurrentNodeIndirectArgs = GraphBuilder.CreateSRV(CurrentIndirectArgs);
					PassParameters->NextNodeIndirectArgs = GraphBuilder.CreateUAV(NextIndirectArgs);
					PassParameters->IndirectArgs = CurrentIndirectArgs;
					PassParameters->NodeLevel = NodeLevel;

					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("NodeCull_%d", NodeLevel),
						ComputeShader,
						PassParameters,
						CurrentIndirectArgs,
						NodeLevel * NANITE_NODE_CULLING_ARG_COUNT * sizeof(uint32)
					);
				}
			}

			// Cluster culling pass
			{
				FRDGBufferRef ClusterCullArgs = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc(3), TEXT("Nanite.ClusterCullArgs"));
				AddPass_InitClusterCullArgs(GraphBuilder, ShaderMap, RDG_EVENT_NAME("InitClusterCullArgs"), QueueParameters.QueueState, ClusterCullArgs, 0);

				FNaniteRayTracingGeometryClusterCutTraversalCS::FPermutationDomain PermutationVector;
				PermutationVector.Set<FNaniteRayTracingGeometryClusterCutTraversalCS::FCullingTypeDim>(NANITE_CULLING_TYPE_CLUSTERS);
				auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingGeometryClusterCutTraversalCS>(PermutationVector);

				auto* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingGeometryClusterCutTraversalCS::FParameters>(&SharedParameters);
				PassParameters->IndirectArgs = ClusterCullArgs;

				PassParameters->PageDataBuffer = GraphBuilder.CreateSRV(PageDataBufferRDG);
				PassParameters->CLASDataBuffer = GraphBuilder.CreateSRV(CLASDataBufferRDG);

				PassParameters->RWBLASBuildDescs = GraphBuilder.CreateUAV(BLASBuildBuffer);
				PassParameters->RWCLASAddresses = GraphBuilder.CreateUAV(CLASAddressesBuffer);

				PassParameters->BLASDataBuffer = GraphBuilder.CreateSRV(BLASDataBufferRDG);

				FComputeShaderUtils::AddPass(
					GraphBuilder,
					RDG_EVENT_NAME("ClusterCull"),
					ComputeShader,
					PassParameters,
					ClusterCullArgs,
					0
				);
			}
		}

		{
			FRDGBufferDesc BLASSizesBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), Geometries.Num());
			FRDGBufferRef BLASSizesBuffer = GraphBuilder.CreateBuffer(BLASSizesBufferDesc, TEXT("NaniteRayTracing.BLASSizesBuffer"));

			FRDGBufferDesc BLASAddressesBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(GPU_VIRTUAL_ADDRESS), Geometries.Num());
			FRDGBufferRef BLASAddressesBuffer = GraphBuilder.CreateBuffer(BLASAddressesBufferDesc, TEXT("NaniteRayTracing.BLASAddressesBuffer"));

			{
				FNaniteRayTracingBuildBLASesParams* PassParams = GraphBuilder.AllocParameters<FNaniteRayTracingBuildBLASesParams>();
				PassParams->ScratchBuffer = StreamOutBuffers->ScratchBuffer;
				PassParams->BLASCountBuffer = BLASCountBuffer;
				PassParams->DescriptorsBuffer = BLASBuildBuffer;
				PassParams->BLASSizesBuffer = BLASSizesBuffer;
				PassParams->BLASAddressesBuffer = BLASAddressesBuffer;
				PassParams->BLASStagingBuffer = BLASStagingBufferRDG;
				PassParams->CLASAddressesBuffer = CLASAddressesBuffer;
				PassParams->CLASBuffer = CLASBufferRDG;

				GraphBuilder.AddPass(RDG_EVENT_NAME("NaniteRayTracing::BuildBLASes"), PassParams, ERDGPassFlags::Compute,
					[PassParams, BLASOpInitializer](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
					{
						FRayTracingClusterOperationParams OpParams = {};
						OpParams.Initializer = BLASOpInitializer;

						OpParams.Resources.In.Scratch = PassParams->ScratchBuffer->GetRHI();
						OpParams.Resources.In.Descriptors = PassParams->DescriptorsBuffer->GetRHI();
						OpParams.Resources.In.ResultCount = PassParams->BLASCountBuffer->GetRHI();

						OpParams.Resources.InOut.Addresses = PassParams->BLASAddressesBuffer->GetRHI();
						OpParams.Resources.Out.Sizes = PassParams->BLASSizesBuffer->GetRHI();
						OpParams.Resources.Out.AccelerationStructures = PassParams->BLASStagingBuffer->GetRHI();

						OpParams.Resources.In.AdditionalResources.Add(PassParams->CLASAddressesBuffer->GetRHI());
						OpParams.Resources.In.AdditionalResources.Add(PassParams->CLASBuffer->GetRHI());

						RHICmdList.ExecuteMultiIndirectClusterOperation(OpParams);
					});
			}

			FRDGBufferRef MoveCountBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 1), TEXT("NaniteRayTracing.BLAS.MoveCountBuffer"));
			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(MoveCountBuffer), 0);

			FRDGBufferDesc AddressesBufferDesc = FRDGBufferDesc::CreateStructuredDesc(sizeof(GPU_VIRTUAL_ADDRESS), Geometries.Num());
			FRDGBufferRef SrcAddressesBuffer = GraphBuilder.CreateBuffer(AddressesBufferDesc, TEXT("NaniteRayTracing.BLAS.SrcAddressesBuffer"));
			FRDGBufferRef DstAddressesBuffer = GraphBuilder.CreateBuffer(AddressesBufferDesc, TEXT("NaniteRayTracing.BLAS.DstAddressesBuffer"));

			{
				FNaniteRayTracingAllocateBLASCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingAllocateBLASCS::FParameters>();
				PassParameters->BLASSizes = GraphBuilder.CreateSRV(BLASSizesBuffer);
				PassParameters->BLASAddresses = GraphBuilder.CreateSRV(BLASAddressesBuffer);
				PassParameters->NumBLAS = Geometries.Num();
				PassParameters->ResourceAddresses = GraphBuilder.CreateSRV(ResourceAddressesBufferRDG);
				PassParameters->BLASAlignment = GRHIGlobals.RayTracing.AccelerationStructureAlignment;
				PassParameters->RWMoveCount = GraphBuilder.CreateUAV(MoveCountBuffer);
				PassParameters->RWSrcAddresses = GraphBuilder.CreateUAV(SrcAddressesBuffer);
				PassParameters->RWDstAddresses = GraphBuilder.CreateUAV(DstAddressesBuffer);
				PassParameters->CutRequests = GraphBuilder.CreateSRV(RequestBuffer);
				PassParameters->RWBLASData = GraphBuilder.CreateUAV(BLASDataBufferRDG);
				PassParameters->RWBLASCacheRequest = GraphBuilder.CreateUAV(BLASCacheRequestBuffer);
				PassParameters->BLASCacheAllocationTable = BLASCacheAllocationTableSRV;
				PassParameters->RWBLASCacheMetadata = GraphBuilder.CreateUAV(BLASCacheMetadataBufferRDG);
				PassParameters->BLASCacheEnabled = bCacheEnabled ? 1 : 0;

				FNaniteRayTracingAllocateBLASCS::FPermutationDomain PermutationVector;
				SetStatsArgsAndPermutation<FNaniteRayTracingAllocateBLASCS>(StatsBufferUAV, PassParameters, PermutationVector);

				auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingAllocateBLASCS>(PermutationVector);
				FIntVector GroupCount = FComputeShaderUtils::GetGroupCountWrapped(Geometries.Num(), FNaniteRayTracingAllocateBLASCS::ThreadGroupSize);

				FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::AllocateBLAS"), ComputeShader, PassParameters, GroupCount);

				if (bCacheEnabled)
				{
					BLASCache->SubmitRequestBuffer(GraphBuilder, BLASCacheRequestBuffer, NumRootPages);
				}
			}

			// Move cache-bound BLASes from staging to cache buffer
			{
				FRayTracingClusterOperationInitializer MoveOpInitializer = {};
				MoveOpInitializer.MaxResultCount = Geometries.Num();
				MoveOpInitializer.Type = ERayTracingClusterOperationType::MOVE;
				MoveOpInitializer.Mode = ERayTracingClusterOperationMode::EXPLICIT_DESTINATIONS;
				MoveOpInitializer.Flags = ERayTracingClusterOperationFlags::NONE;
				MoveOpInitializer.Operation.Move.Type = ERayTracingClusterOperationMoveType::BOTTOM_LEVEL;
				MoveOpInitializer.Operation.Move.MaxBytes = BLASOpSize.ResultMaxSizeInBytes;

				FRayTracingClusterOperationSize MoveOpSize = RHICalcRayTracingClusterOperationSize(MoveOpInitializer);
				StreamOutBuffers->RequestScratch(GraphBuilder, MoveOpSize.ScratchSizeInBytes);

				FNaniteRayTracingMoveASesParams* PassParams = GraphBuilder.AllocParameters<FNaniteRayTracingMoveASesParams>();
				PassParams->ScratchBuffer = StreamOutBuffers->ScratchBuffer;
				PassParams->CountBuffer = MoveCountBuffer;
				PassParams->SrcAddressesBuffer = SrcAddressesBuffer;
				PassParams->DstAddressesBuffer = DstAddressesBuffer;
				PassParams->StagingBuffer = BLASStagingBufferRDG;
				PassParams->DstBuffer = BLASCacheBufferRDG;

				GraphBuilder.AddPass(RDG_EVENT_NAME("NaniteRayTracing::MoveBLASes"), PassParams, ERDGPassFlags::Compute,
					[PassParams, MoveOpInitializer](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
					{
						FRayTracingClusterOperationParams OpParams = {};
						OpParams.Initializer = MoveOpInitializer;

						OpParams.Resources.In.Scratch = PassParams->ScratchBuffer->GetRHI();
						OpParams.Resources.In.Descriptors = PassParams->SrcAddressesBuffer->GetRHI();
						OpParams.Resources.In.ResultCount = PassParams->CountBuffer->GetRHI();

						OpParams.Resources.InOut.Addresses = PassParams->DstAddressesBuffer->GetRHI();

						OpParams.Resources.In.AdditionalResources.Add(PassParams->StagingBuffer->GetRHI());
						OpParams.Resources.In.AdditionalResources.Add(PassParams->DstBuffer->GetRHI());

						RHICmdList.ExecuteMultiIndirectClusterOperation(OpParams);
					});
			}

			INC_DWORD_STAT_BY(STAT_NaniteRayTracingBLASBuilds, Geometries.Num());
		}

	}

	void FRayTracingManager::DebugModeCLAS(FRDGBuilder& GraphBuilder)
	{
		if (!CVarNaniteRayTracingDebug.GetValueOnRenderThread())
		{
			return;
		}

		RDG_EVENT_SCOPE(GraphBuilder, "NaniteRayTracing::DebugModeCLAS");

		ShaderPrint::SetEnabled(true);

		//ue-todo: Refactor to use EShaderPlatform directly instead of converting from FeatureLevel
		FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GShaderPlatformForFeatureLevel[GetFeatureLevel()]);

		const bool bShowHistograms = CVarNaniteRayTracingDebugHistograms.GetValueOnRenderThread();
		const uint32 HistogramSize = 32;

		FRDGBufferRef CLASHistogramBuffer;
		FRDGBufferRef CLASSizeSumBuffer;
		FRDGBufferRef BLASHistogramBuffer;

		if (bShowHistograms)
		{
			CLASHistogramBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), HistogramSize), TEXT("CLASHistogram"));
			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(CLASHistogramBuffer), 0);

			CLASSizeSumBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 1), TEXT("CLASSizeSum"));
			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(CLASSizeSumBuffer), 0);

			FRDGBufferRef CLASDataBufferRDG = CLASDataBuffer ? GraphBuilder.RegisterExternalBuffer(CLASDataBuffer) : GSystemTextures.GetDefaultByteAddressBuffer(GraphBuilder, sizeof(FNaniteRayTracingASData));

			{
				FNaniteRayTracingDebugHistogramCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingDebugHistogramCS::FParameters>();

				PassParameters->ASDataBuffer = GraphBuilder.CreateSRV(CLASDataBufferRDG);
				PassParameters->NumAS = CLASDataAllocator.GetMaxSize();

				PassParameters->RWHistogram = GraphBuilder.CreateUAV(CLASHistogramBuffer);
				PassParameters->HistogramScaleBias = FVector2f(0, 0);
				PassParameters->HistogramSize = HistogramSize;

				PassParameters->RWSizeSum = GraphBuilder.CreateUAV(CLASSizeSumBuffer);

				FNaniteRayTracingDebugHistogramCS::FPermutationDomain PermutationVector;

				auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingDebugHistogramCS>(PermutationVector);
				FIntVector GroupCount = FComputeShaderUtils::GetGroupCountWrapped(CLASDataAllocator.GetMaxSize(), FNaniteRayTracingDebugHistogramCS::ThreadGroupSize);

				FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::DebugHistogramCLAS"), ComputeShader, PassParameters, GroupCount);
			}

			BLASHistogramBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), HistogramSize), TEXT("BLASHistogram"));
			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(BLASHistogramBuffer), 0);

			FRDGBufferRef BLASSizeSumBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 1), TEXT("BLASSizeSum"));
			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(BLASSizeSumBuffer), 0);

			FRDGBufferRef BLASDataBufferRDG = BLASDataBuffer ? GraphBuilder.RegisterExternalBuffer(BLASDataBuffer) : GSystemTextures.GetDefaultByteAddressBuffer(GraphBuilder, sizeof(FNaniteRayTracingASData));

			{
				FNaniteRayTracingDebugHistogramCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingDebugHistogramCS::FParameters>();

				PassParameters->ASDataBuffer = GraphBuilder.CreateSRV(BLASDataBufferRDG);
				PassParameters->NumAS = MaxRootPageIndex + 1;

				PassParameters->RWHistogram = GraphBuilder.CreateUAV(BLASHistogramBuffer);
				PassParameters->HistogramScaleBias = FVector2f(0, 0);
				PassParameters->HistogramSize = HistogramSize;

				PassParameters->RWSizeSum = GraphBuilder.CreateUAV(BLASSizeSumBuffer);

				FNaniteRayTracingDebugHistogramCS::FPermutationDomain PermutationVector;

				auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingDebugHistogramCS>(PermutationVector);
				FIntVector GroupCount = FComputeShaderUtils::GetGroupCountWrapped(MaxRootPageIndex + 1, FNaniteRayTracingDebugHistogramCS::ThreadGroupSize);

				FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::DebugHistogramBLAS"), ComputeShader, PassParameters, GroupCount);
			}
		}
		else
		{
			CLASHistogramBuffer = GSystemTextures.GetDefaultStructuredBuffer(GraphBuilder, sizeof(uint32));
			CLASSizeSumBuffer = GSystemTextures.GetDefaultStructuredBuffer(GraphBuilder, sizeof(uint32));
			BLASHistogramBuffer = GSystemTextures.GetDefaultStructuredBuffer(GraphBuilder, sizeof(uint32));
		}

		FRDGBufferRef CLASAllocatorBufferRDG = CLASAllocatorBuffer ? GraphBuilder.RegisterExternalBuffer(CLASAllocatorBuffer) : GSystemTextures.GetDefaultStructuredBuffer(GraphBuilder, sizeof(uint32), 0u);

		{
			FNaniteRayTracingDebugCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNaniteRayTracingDebugCS::FParameters>();
			ShaderPrint::SetParameters(GraphBuilder, PassParameters->ShaderPrint);

			PassParameters->CLASPoolAllocationSize = CLASPoolAllocationSize;
			PassParameters->CLASPoolSize = CLASPoolSize;
			PassParameters->CLASBufferSize = CLASBuffer->GetSize();
			PassParameters->NumCLAS = CLASDataAllocator.GetSparselyAllocatedSize();
			PassParameters->CLASAllocator = GraphBuilder.CreateSRV(CLASAllocatorBufferRDG);
			PassParameters->CLASHistogram = GraphBuilder.CreateSRV(CLASHistogramBuffer);
			PassParameters->CLASSizeSum = GraphBuilder.CreateSRV(CLASSizeSumBuffer);

			PassParameters->BLASCacheBufferSize = BLASCache ? BLASCache->GetBuffer()->GetSize() : 0;
			PassParameters->BLASStagingBufferSize = BLASStagingBuffer ? BLASStagingBuffer->GetSize() : 0;
			PassParameters->NumBLAS = Geometries.Num();
			PassParameters->BLASHistogram = GraphBuilder.CreateSRV(BLASHistogramBuffer);

			PassParameters->StatsBuffer = GraphBuilder.CreateSRV(StatsBufferRDG ? StatsBufferRDG : GSystemTextures.GetDefaultStructuredBuffer(GraphBuilder, sizeof(uint32), 0u));

			PassParameters->ShowHistograms = bShowHistograms ? 1 : 0;

			FNaniteRayTracingDebugCS::FPermutationDomain PermutationVector;

			auto ComputeShader = ShaderMap->GetShader<FNaniteRayTracingDebugCS>(PermutationVector);
			FIntVector GroupCount = FIntVector(1, 1, 1);

			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NaniteRayTracing::Debug"), ComputeShader, PassParameters, GroupCount);
		}
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FNaniteRayTracingPrimitivesParams, )
		RDG_BUFFER_ACCESS(ScratchBuffer, ERHIAccess::UAVCompute)
	END_SHADER_PARAMETER_STRUCT()

	bool FRayTracingManager::ProcessBuildRequests(FRDGBuilder& GraphBuilder)
	{
		if (!bInitialized)
		{
			return false;
		}

		RDG_EVENT_SCOPE_STAT(GraphBuilder, NaniteRayTracingProcessBuildRequests, "NaniteRayTracing::ProcessBuildRequests");

		// resize AuxiliaryDataBuffer if necessary
		FRDGBufferRef AuxiliaryDataBufferRDG;
		{
			uint32 MinAuxiliaryBufferEntries;
			EAllowShrinking AllowShrinking;

			if (GetRayTracingMode() == ERayTracingMode::Fallback)
			{
				// when not using Nanite Ray Tracing allow AuxiliaryDataBuffer to shrink to initial size 
				MinAuxiliaryBufferEntries = GDisabledMinAuxiliaryBufferEntries;
				AllowShrinking = EAllowShrinking::Yes;
			}
			else
			{
				MinAuxiliaryBufferEntries = GMinAuxiliaryBufferEntries;
				AllowShrinking = EAllowShrinking::No;
			}

			const uint32 NumAuxiliaryDataEntries = FMath::Max((uint32)AuxiliaryDataAllocator.GetMaxSize(), MinAuxiliaryBufferEntries);
			AuxiliaryDataBufferRDG = ResizeByteAddressBufferIfNeeded(GraphBuilder, AuxiliaryDataBuffer, GetAuxiliaryEntrySize() * NumAuxiliaryDataEntries, TEXT("NaniteRayTracing.AuxiliaryDataBuffer"), /*bCopy*/ true, AllowShrinking);

			SET_MEMORY_STAT(STAT_NaniteRayTracingAuxiliaryDataBuffer, AuxiliaryDataBufferRDG->GetSize());
		}

		FRDGBufferRef StagingAuxiliaryDataBufferRDG = ScheduledBuilds.IsEmpty() ? nullptr : GraphBuilder.RegisterExternalBuffer(StagingAuxiliaryDataBuffer);

		TArray<FRayTracingGeometryBuildParams> BuildParams;
		uint32 BLASScratchSize = 0;

		const uint32 AuxiliaryEntrySize = GetAuxiliaryEntrySize();
		
		for (uint32 GeometryId : ScheduledBuilds)
		{
			FInternalData& Data = *Geometries[GeometryId];

			const FRayTracingGeometryInitializer& Initializer = Data.RayTracingGeometryRHI->GetInitializer();

			FRayTracingGeometryBuildParams Params;
			Params.Geometry = Data.RayTracingGeometryRHI;
			Params.BuildMode = EAccelerationStructureBuildMode::Build;

			BuildParams.Add(Params);

			FRayTracingAccelerationStructureSize SizeInfo = RHICalcRayTracingGeometrySize(Initializer);
			BLASScratchSize = AlignArbitrary(BLASScratchSize + SizeInfo.BuildScratchSize, GRHIGlobals.RayTracing.ScratchBufferAlignment);

			Data.bUpdating = false;
			Data.BaseMeshDataOffset = -1;

			DEC_DWORD_STAT_BY(STAT_NaniteRayTracingInFlightUpdates, 1);

			// copy from staging to persistent auxiliary data buffer
			AddCopyBufferPass(GraphBuilder, AuxiliaryDataBufferRDG, Data.AuxiliaryDataOffset * AuxiliaryEntrySize, StagingAuxiliaryDataBufferRDG, Data.StagingAuxiliaryDataOffset * AuxiliaryEntrySize, Data.AuxiliaryDataSize * AuxiliaryEntrySize);
			Data.StagingAuxiliaryDataOffset = INDEX_NONE;
		}

		const uint32 BLASScratchSizeMultiple = FMath::Max(GNaniteRayTracingBLASScratchSizeMultipleMB, 1) * 1024 * 1024;
		BLASScratchSize = FMath::DivideAndRoundUp(BLASScratchSize, BLASScratchSizeMultiple) * BLASScratchSizeMultiple;

		INC_DWORD_STAT_BY(STAT_NaniteRayTracingScheduledBuilds, ScheduledBuilds.Num());

		ScheduledBuilds.Empty();
		ScheduledBuildsNumPrimitives = 0;

		bool bAnyBlasRebuilt = false;

		if (BuildParams.Num() > 0)
		{
			RDG_EVENT_SCOPE_STAT(GraphBuilder, NaniteRayTracingRebuildBLAS, "NaniteRayTracingRebuildBLAS");

			FRDGBufferDesc ScratchBufferDesc;
			ScratchBufferDesc.Usage = EBufferUsageFlags::RayTracingScratch | EBufferUsageFlags::StructuredBuffer;
			ScratchBufferDesc.BytesPerElement = GRHIGlobals.RayTracing.ScratchBufferAlignment;
			ScratchBufferDesc.NumElements = FMath::DivideAndRoundUp(BLASScratchSize, ScratchBufferDesc.BytesPerElement);

			FRDGBufferRef ScratchBuffer = GraphBuilder.CreateBuffer(ScratchBufferDesc, TEXT("NaniteRayTracing.BLASSharedScratchBuffer"));

			FNaniteRayTracingPrimitivesParams* PassParams = GraphBuilder.AllocParameters<FNaniteRayTracingPrimitivesParams>();
			PassParams->ScratchBuffer = ScratchBuffer;

			GraphBuilder.AddPass(RDG_EVENT_NAME("NaniteRayTracing::UpdateBLASes"), PassParams, ERDGPassFlags::Compute | ERDGPassFlags::NeverCull,
				[PassParams, BuildParams = MoveTemp(BuildParams)](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
			{
				FRHIBufferRange ScratchBufferRange;
				ScratchBufferRange.Buffer = PassParams->ScratchBuffer->GetRHI();
				ScratchBufferRange.Offset = 0;
				RHICmdList.BuildAccelerationStructures(BuildParams, ScratchBufferRange);
			});

			bAnyBlasRebuilt = true;
		}

		if (ReadbackBuffersNumPending == 0 && PendingBuilds.IsEmpty())
		{
			bUpdating = false;
		}

		return bAnyBlasRebuilt;
	}

	FRHIRayTracingGeometry* FRayTracingManager::GetRayTracingGeometry(const FPrimitiveSceneInfo* SceneInfo) const
	{
		auto NaniteProxy = static_cast<const Nanite::FSceneProxyBase*>(SceneInfo->Proxy);

		const uint32 Id = NaniteProxy->GetRayTracingId();

		if (Id == INDEX_NONE)
		{
			return nullptr;
		}

		const FInternalData* Data = Geometries[Id];

		return Data->RayTracingGeometryRHI;
	}

	uint32 FRayTracingManager::GetRayTracingGeometrySegmentCount(const FPrimitiveSceneInfo* SceneInfo) const
	{
		auto NaniteProxy = static_cast<const Nanite::FSceneProxyBase*>(SceneInfo->Proxy);

		const uint32 Id = NaniteProxy->GetRayTracingId();

		if (Id == INDEX_NONE)
		{
			return 0;
		}

		return Geometries[Id]->NumSegments;
	}


	void FRayTracingManager::GetTLASBuildDependencies(FRDGBuilder& GraphBuilder, FRDGBufferAccessArray& OutBufferAccesses) const
	{
		if (GetRayTracingMode() != ERayTracingMode::CLAS)
		{
			return;
		}

		if (BLASStagingBuffer)
		{
			OutBufferAccesses.Emplace(GraphBuilder.RegisterExternalBuffer(BLASStagingBuffer), ERHIAccess::BVHRead);
		}
		if (BLASCache)
		{
			OutBufferAccesses.Emplace(GraphBuilder.RegisterExternalBuffer(BLASCache->GetBuffer()), ERHIAccess::BVHRead);
		}
		if (CLASBuffer)
		{
			OutBufferAccesses.Emplace(GraphBuilder.RegisterExternalBuffer(CLASBuffer), ERHIAccess::BVHRead);
		}
	}

	bool FRayTracingManager::CheckModeChanged()
	{
		bPrevMode = bCurrentMode;
		bCurrentMode = GetRayTracingMode();
		return bPrevMode != bCurrentMode;
	}

	void FRayTracingManager::PostRender()
	{
		VisibleGeometries.Empty();
		VisiblePrimitives.Empty();
	}

	void FRayTracingManager::EndFrame()
	{
		// clear RDG resources since they can't be reused over multiple frames
		StatsBufferRDG = nullptr;
		StatsBufferUAV = nullptr;
		UniformBuffer = nullptr;
	}

	void FRayTracingManager::UpdateUniformBuffer(FRDGBuilder& GraphBuilder, bool bShouldRenderNanite)
	{
		FNaniteRayTracingUniformParameters* Parameters = GraphBuilder.AllocParameters<FNaniteRayTracingUniformParameters>();

		if (bShouldRenderNanite && bCurrentMode != ERayTracingMode::Fallback)
		{
			Parameters->PageConstants.X = 0;
			Parameters->PageConstants.Y = Nanite::GStreamingManager.GetMaxStreamingPages();
			Parameters->MaxNodes = Nanite::FGlobalResources::GetMaxNodes();
			Parameters->ClusterPageData = Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
			Parameters->HierarchyBuffer = Nanite::GStreamingManager.GetHierarchySRV(GraphBuilder);
			Parameters->RayTracingDataBuffer = Nanite::GRayTracingManager.GetAuxiliaryDataSRV(GraphBuilder);
			Parameters->UsingCLAS = bCurrentMode == ERayTracingMode::CLAS ? 1 : 0;
		}
		else
		{
			Parameters->PageConstants.X = 0;
			Parameters->PageConstants.Y = 0;
			Parameters->MaxNodes = 0;
			Parameters->ClusterPageData = GraphBuilder.CreateSRV(GSystemTextures.GetDefaultByteAddressBuffer(GraphBuilder, 4u));
			Parameters->HierarchyBuffer = GraphBuilder.CreateSRV(GSystemTextures.GetDefaultByteAddressBuffer(GraphBuilder, 4u));
			Parameters->RayTracingDataBuffer = GraphBuilder.CreateSRV(GSystemTextures.GetDefaultByteAddressBuffer(GraphBuilder, 8u));
			Parameters->UsingCLAS = 0;
		}

		UniformBuffer = GraphBuilder.CreateUniformBuffer(Parameters);
	}

	TGlobalResource<FRayTracingManager> GRayTracingManager;
} // namespace Nanite

FAutoConsoleVariableDeprecated CVarNaniteRayTracingCutError(TEXT("r.RayTracing.Nanite.CutError"), TEXT("r.RayTracing.Nanite.MinCutError"), TEXT("5.8"));
FAutoConsoleVariableDeprecated CVarNaniteRayTracingStreaming(TEXT("r.RayTracing.Nanite.Streaming"), TEXT("r.RayTracing.Nanite.DriveStreaming"), TEXT("5.8"));
FAutoConsoleVariableDeprecated CVarNaniteRayTracingStreamingLodBias(TEXT("r.RayTracing.Nanite.Streaming.LODBias"), TEXT("r.RayTracing.Nanite.LODBias"), TEXT("5.8"));
FAutoConsoleVariableDeprecated CVarNaniteRayTracingStreamingOffscreenLodBias(TEXT("r.RayTracing.Nanite.Streaming.Offscreen.LODBias"), TEXT("r.RayTracing.Nanite.Offscreen.LODBias"), TEXT("5.8"));
FAutoConsoleVariableDeprecated CVarNaniteRayTracingStreamingOffscreenMinCutError(TEXT("r.RayTracing.Nanite.Streaming.Offscreen.MinCutError"), TEXT("r.RayTracing.Nanite.Offscreen.MinCutError"), TEXT("5.8"));

#endif // RHI_RAYTRACING
