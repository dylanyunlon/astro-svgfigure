// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "HAL/Platform.h"
#include "RHIDeviceAnalytics.h"

namespace RHI::DeviceAnalytics
{

// Important!
// Update this number every time meaning of enums change in defines below.
// No need to update if values are appended without changing meaning of existing values.
static const uint16 RHIDeviceAnalyticsBaseSchemaVersion = 1;

// GRHIGlobals
// visitor gets a full name of a field due to many nested fields
#define RHI_DEVICE_ANALYTICS_KEYS_GLOBALS(M, C) \
	M(C, RHIG_, Integrated,            DeviceIsIntegrated,                                            1) \
	M(C, RHIG_, Uma,                   DeviceIsCacheCoherentUMA,                                      2) \
	M(C, RHIG_, UmaSepVa,              DeviceIsUMASeparateVA,                                         3) \
	M(C, RHIG_, GpuUploadHeaps,        DeviceSupportsGPUUploadHeaps,                                  4) \
	M(C, RHIG_, NullRhi,               UsingNullRHI,                                                  5) \
	M(C, RHIG_, Bindless,              bSupportsBindless,                                             6) \
	M(C, RHIG_, BindlessDescRng,       bSupportsDescriptorRange,                                      7) \
	M(C, RHIG_, WaveOps,               SupportsWaveOperations,                                        8) \
	M(C, RHIG_, WaveMin,               MinimumWaveSize,                                               9) \
	M(C, RHIG_, WaveMax,               MaximumWaveSize,                                               10) \
	M(C, RHIG_, Native16,              SupportsNative16BitOps,                                        11) \
	M(C, RHIG_, MsT0,                  SupportsMeshShadersTier0,                                      12) \
	M(C, RHIG_, MsT1,                  SupportsMeshShadersTier1,                                      13) \
	M(C, RHIG_, WgT1,                  SupportsShaderWorkGraphsTier1,                                 14) \
	M(C, RHIG_, WgT1_1,                SupportsShaderWorkGraphsTier1_1,                               15) \
	M(C, RHIG_, Barycentrics,          SupportsBarycentricsSemantic,                                  16) \
	M(C, RHIG_, Ser,                   SupportsShaderExecutionReordering,                             17) \
	M(C, RHIG_, ShaderTimestamp,       SupportsShaderTimestamp,                                       18) \
	M(C, RHIG_, ShaderRootConst,       SupportsShaderRootConstants,                                   19) \
	M(C, RHIG_, StencilRefPs,          SupportsStencilRefFromPixelShader,                             20) \
	M(C, RHIG_, RasterOrderView,       SupportsRasterOrderViews,                                      21) \
	M(C, RHIG_, ConservativeRast,      SupportsConservativeRasterization,                             22) \
	M(C, RHIG_, PrimShaders,           SupportsPrimitiveShaders,                                      23) \
	M(C, RHIG_, AtomicU64,             SupportsAtomicUInt64,                                          24) \
	M(C, RHIG_, ArrayIdxAnySh,         SupportsArrayIndexFromAnyShader,                               25) \
	M(C, RHIG_, Rt,                    RayTracing.Supported,                                          26) \
	M(C, RHIG_, RtShaders,             RayTracing.SupportsShaders,                                    27) \
	M(C, RHIG_, RtInline,              RayTracing.SupportsInlineRayTracing,                           28) \
	M(C, RHIG_, RtInlineSbt,           RayTracing.RequiresInlineRayTracingSBT,                        29) \
	M(C, RHIG_, RtInlinedCb,           RayTracing.SupportsInlinedCallbacks,                           30) \
	M(C, RHIG_, RtPsoAdd,              RayTracing.SupportsPSOAdditions,                               31) \
	M(C, RHIG_, RtDispIndirect,        RayTracing.SupportsDispatchIndirect,                           32) \
	M(C, RHIG_, RtAsyncBuild,          RayTracing.SupportsAsyncBuildAccelerationStructure,            33) \
	M(C, RHIG_, RtTlasIndirect,        RayTracing.SupportsBuildTopLevelAccelerationStructureIndirect, 34) \
	M(C, RHIG_, RtAsyncDisp,           RayTracing.SupportsAsyncRayTraceDispatch,                      35) \
	M(C, RHIG_, RtClusterOps,          RayTracing.SupportsClusterOps,                                 36) \
	M(C, RHIG_, RtAmdHitToken,         RayTracing.SupportsAMDHitToken,                                37) \
	M(C, RHIG_, RtPersistentSbt,       RayTracing.SupportsPersistentSBTs,                             38) \
	M(C, RHIG_, RtLooseParams,         RayTracing.SupportsLooseParamsInShaderRecord,                  39) \
	M(C, RHIG_, RtAsCompact,           RayTracing.SupportsAccelerationStructureCompaction,            40) \
	M(C, RHIG_, RtAsSerialize,         RayTracing.SupportsSerializeAccelerationStructure,             41) \
	M(C, RHIG_, RtSepHitGroup,         RayTracing.RequiresSeparateHitGroupContributionsBuffer,        42) \
	M(C, RHIG_, RtLss,                 RayTracing.SupportsLinearSweptSpheres,                         43) \
	M(C, RHIG_, RtAsAlign,             RayTracing.AccelerationStructureAlignment,                     44) \
	M(C, RHIG_, RtClasAlign,           RayTracing.ClusterAccelerationStructureAlignment,              45) \
	M(C, RHIG_, RtClasTplAlign,        RayTracing.ClusterAccelerationStructureTemplateAlignment,      46) \
	M(C, RHIG_, RtScratchAlign,        RayTracing.ScratchBufferAlignment,                             47) \
	M(C, RHIG_, RtSbtAlign,            RayTracing.ShaderTableAlignment,                               48) \
	M(C, RHIG_, RtInstDescSz,          RayTracing.InstanceDescriptorSize,                             49) \
	M(C, RHIG_, VrsPipe,               VariableRateShading.SupportsPipeline,                          50) \
	M(C, RHIG_, VrsAttach,             VariableRateShading.SupportsAttachment,                        51) \
	M(C, RHIG_, VrsLarger,             VariableRateShading.SupportsLargerSizes,                       52) \
	M(C, RHIG_, VrsComplexComb,        VariableRateShading.SupportsComplexCombinerOps,                53) \
	M(C, RHIG_, VrsArrayAttach,        VariableRateShading.SupportsAttachmentArrayTextures,           54) \
	M(C, RHIG_, VrsLateUpdate,         VariableRateShading.SupportsLateUpdate,                        55) \
	M(C, RHIG_, VrsImageOffsets,       VariableRateShading.SupportsImageOffsets,                      56) \
	M(C, RHIG_, VrsTileMinW,           VariableRateShading.ImageTileMinWidth,                         57) \
	M(C, RHIG_, VrsTileMinH,           VariableRateShading.ImageTileMinHeight,                        58) \
	M(C, RHIG_, VrsTileMaxW,           VariableRateShading.ImageTileMaxWidth,                         59) \
	M(C, RHIG_, VrsTileMaxH,           VariableRateShading.ImageTileMaxHeight,                        60) \
	M(C, RHIG_, ReservedRes,           ReservedResources.Supported,                                   61) \
	M(C, RHIG_, ReservedVol,           ReservedResources.SupportsVolumeTextures,                      62) \
	M(C, RHIG_, SbDispatch,            ShaderBundles.SupportsDispatch,                                63) \
	M(C, RHIG_, SbWgDispatch,          ShaderBundles.SupportsWorkGraphDispatch,                       64) \
	M(C, RHIG_, SbWgGfx,               ShaderBundles.SupportsWorkGraphGraphicsDispatch,               65) \
	M(C, RHIG_, SbParallel,            ShaderBundles.SupportsParallel,                                66) \
	M(C, RHIG_, SbSharedBind,          ShaderBundles.RequiresSharedBindlessParameters,                67) \
	M(C, RHIG_, FbFetch,               SupportsShaderFramebufferFetch,                                68) \
	M(C, RHIG_, FbFetchMrt,            SupportsShaderMRTFramebufferFetch,                             69) \
	M(C, RHIG_, FbFetchPb,             SupportsShaderFramebufferFetchProgrammableBlending,            70) \
	M(C, RHIG_, Pls,                   SupportsPixelLocalStorage,                                     71) \
	M(C, RHIG_, DsFetch,               SupportsShaderDepthStencilFetch,                               72) \
	M(C, RHIG_, MobileMultiView,       SupportsMobileMultiView,                                       73) \
	M(C, RHIG_, ImageExternal,         SupportsImageExternal,                                         74) \
	M(C, RHIG_, LossyFbCompress,       SupportsLossyFramebufferCompression,                           75) \
	M(C, RHIG_, WideMrt,               SupportsWideMRT,                                               76) \
	M(C, RHIG_, MsaaShResolve,         SupportsMSAAShaderResolve,                                     77) \
	M(C, RHIG_, DsResolve,             SupportsDepthStencilResolve,                                   78) \
	M(C, RHIG_, MsaaDepthSample,       SupportsMSAADepthSampleAccess,                                 79) \
	M(C, RHIG_, BbCustomDs,            SupportsBackBufferWithCustomDepthStencil,                      80) \
	M(C, RHIG_, HdrOutput,             SupportsHDROutput,                                             81) \
	M(C, RHIG_, TexViews,              SupportsTextureViews,                                          82) \
	M(C, RHIG_, RawViewsAny,           SupportsRawViewsForAnyBuffer,                                  83) \
	M(C, RHIG_, UavFmtAlias,           SupportsUAVFormatAliasing,                                     84) \
	M(C, RHIG_, DirectGpuLock,         SupportsDirectGPUMemoryLock,                                   85) \
	M(C, RHIG_, BufTexUpdate,          SupportsUpdateFromBufferTexture,                               86) \
	M(C, RHIG_, MapWriteNoOw,          SupportsMapWriteNoOverwrite,                                   87) \
	M(C, RHIG_, EffUploadCreate,       SupportsEfficientUploadOnResourceCreation,                     88) \
	M(C, RHIG_, Tex3d,                 SupportsTexture3D,                                             89) \
	M(C, RHIG_, Tex3dAniso,            SupportsTexture3DAnisotropicFiltering,                         90) \
	M(C, RHIG_, Tex3dBulk,             UseTexture3DBulkData,                                          91) \
	M(C, RHIG_, RtfG8,                 SupportsRenderTargetFormat_PF_G8,                              92) \
	M(C, RHIG_, RtfFp16,               SupportsRenderTargetFormat_PF_FloatRGBA,                       93) \
	M(C, RHIG_, VolTexRender,          SupportsVolumeTextureRendering,                                94) \
	M(C, RHIG_, RwTexBuffers,          SupportsRWTextureBuffers,                                      95) \
	M(C, RHIG_, BindTexArrSlc,         SupportsBindingTexArrayPerSlice,                               96) \
	M(C, RHIG_, SepBlendState,         SupportsSeparateRenderTargetBlendState,                        97) \
	M(C, RHIG_, DualSrcBlend,          SupportsDualSrcBlending,                                       98) \
	M(C, RHIG_, SepDsCopy,             SupportsSeparateDepthStencilCopyAccess,                        99) \
	M(C, RHIG_, DepthBounds,           SupportsDepthBoundsTest,                                       100) \
	M(C, RHIG_, ExplicitHtile,         SupportsExplicitHTile,                                         101) \
	M(C, RHIG_, ExplicitFmask,         SupportsExplicitFMask,                                         102) \
	M(C, RHIG_, ResummarizeHtile,      SupportsResummarizeHTile,                                      103) \
	M(C, RHIG_, DepthUav,              SupportsDepthUAV,                                              104) \
	M(C, RHIG_, PsUav,                 SupportsPixelShaderUAVs,                                       105) \
	M(C, RHIG_, VsUav,                 SupportsVertexShaderUAVs,                                      106) \
	M(C, RHIG_, ReqRtForPsUav,         RequiresRenderTargetForPixelShaderUAVs,                        107) \
	M(C, RHIG_, LinVolFormat,          SupportLinearTextureVolumeFormat,                              108) \
	M(C, RHIG_, AsyncCompute,          SupportsEfficientAsyncCompute,                                 109) \
	M(C, RHIG_, AsyncCompAlias,        SupportsAsyncComputeTransientAliasing,                         110) \
	M(C, RHIG_, AsyncTexCreate,        SupportsAsyncTextureCreation,                                  111) \
	M(C, RHIG_, AsyncTexStream,        SupportsAsyncTextureStreamOut,                                 112) \
	M(C, RHIG_, TsQueries,             SupportsTimestampRenderQueries,                                113) \
	M(C, RHIG_, GpuTsNoBubble,         SupportsGPUTimestampBubblesRemoval,                            114) \
	M(C, RHIG_, FcNoBubble,            SupportsFrameCyclesBubblesRemoval,                             115) \
	M(C, RHIG_, GpuUsage,              SupportsGPUUsage,                                              116) \
	M(C, RHIG_, HwHsr,                 HardwareHiddenSurfaceRemoval,                                  117) \
	M(C, RHIG_, ExactOcclQ,            SupportsExactOcclusionQueries,                                 118) \
	M(C, RHIG_, ParallelOcclQ,         SupportsParallelOcclusionQueries,                              119) \
	M(C, RHIG_, AsyncGetQuery,         SupportsAsyncGetRenderQueryResult,                             120) \
	M(C, RHIG_, DrawIndirect,          SupportsDrawIndirect,                                          121) \
	M(C, RHIG_, MultiDrawIndir,        SupportsMultiDrawIndirect,                                     122) \
	M(C, RHIG_, BaseVertexIdx,         SupportsBaseVertexIndex,                                       123) \
	M(C, RHIG_, FirstInstance,         SupportsFirstInstance,                                         124) \
	M(C, RHIG_, QuadTopology,          SupportsQuadTopology,                                          125) \
	M(C, RHIG_, RectTopology,          SupportsRectTopology,                                          126) \
	M(C, RHIG_, PipeSortKey,           SupportsPipelineStateSortKey,                                  127) \
	M(C, RHIG_, QuadBufStereo,         SupportsQuadBufferStereo,                                      128) \
	M(C, RHIG_, RdtSr,                 SupportsRenderDepthTargetableShaderResources,                  129) \
	M(C, RHIG_, RhiThread,             SupportsRHIThread,                                             130) \
	M(C, RHIG_, RhiOnTaskThrd,         SupportsRHIOnTaskThread,                                       131) \
	M(C, RHIG_, ParallelRhiExec,       SupportsParallelRHIExecute,                                    132) \
	M(C, RHIG_, ConcurrentTxSub,       SupportsConcurrentTranslateAndSubmit,                          133) \
	M(C, RHIG_, Multithreading,        SupportsMultithreading,                                        134) \
	M(C, RHIG_, MtShaderCreate,        SupportsMultithreadedShaderCreation,                           135) \
	M(C, RHIG_, MtResources,           SupportsMultithreadedResources,                                136) \
	M(C, RHIG_, ParallelRp,            ParallelRenderPasses.Supported,                                137) \
	M(C, RHIG_, ParallelRpCust,        ParallelRenderPasses.UsesCustomContexts,                       138) \
	M(C, RHIG_, RtWithSepRhi,          SupportsParallelRenderingTasksWithSeparateRHIThread,           139) \
	M(C, RHIG_, AsyncPsoPrecmp,        SupportsAsyncPipelinePrecompile,                               140) \
	M(C, RHIG_, SplitBarriers,         SupportsSplitBarriers,                                         141) \
	M(C, RHIG_, RhiNeedsKick,          RHIThreadNeedsKicking,                                         142) \
	M(C, RHIG_, PsoPrecaching,         SupportsPSOPrecaching,                                         143) \
	M(C, RHIG_, PipeFileCache,         SupportsPipelineFileCache,                                     144) \
	M(C, RHIG_, LazyShLoading,         SupportsLazyShaderCodeLoading,                                 145) \
	M(C, RHIG_, DynResolution,         SupportsDynamicResolution,                                     146) \
	M(C, RHIG_, TexStreaming,          SupportsTextureStreaming,                                      147) \
	M(C, RHIG_, NeedsExtraDel,         NeedsExtraDeletionLatency,                                     148) \
	M(C, RHIG_, ForceNoDelTex,         ForceNoDeletionLatencyForStreamingTextures,                    149) \
	M(C, RHIG_, NeedsShUnbinds,        NeedsShaderUnbinds,                                            150) \
	M(C, RHIG_, NeedsXtraTrans,        NeedsExtraTransitions,                                         151) \
	M(C, RHIG_, TdStateTrack,          NeedsTransientDiscardStateTracking,                            152) \
	M(C, RHIG_, TdGfxWorkaround,       NeedsTransientDiscardOnGraphicsWorkaround,                     153) \
	M(C, RHIG_, UnatlCsmWkar,          NeedsUnatlasedCSMDepthsWorkaround,                             154) \
	M(C, RHIG_, SrvGfxNpWkar,          NeedsSRVGraphicsNonPixelWorkaround,                            155) \
	M(C, RHIG_, DediSrvGfxNp,          SupportsDedicatedSRVGraphicsNonPixelAccess,                    156) \
	M(C, RHIG_, DebugLayer,            IsDebugLayerEnabled,                                           157) \
	M(C, RHIG_, MaxTexDim,             MaxTextureDimensions,                                          158) \
	M(C, RHIG_, MaxCubeDim,            MaxCubeTextureDimensions,                                      159) \
	M(C, RHIG_, MaxVolDim,             MaxVolumeTextureDimensions,                                    160) \
	M(C, RHIG_, MaxTexArrLyrs,         MaxTextureArrayLayers,                                         161) \
	M(C, RHIG_, MaxTexSamplers,        MaxTextureSamplers,                                            162) \
	M(C, RHIG_, MaxTexMips,            MaxTextureMipCount,                                            163) \
	M(C, RHIG_, MaxSimulUavs,          MaxSimultaneousUAVs,                                           164) \
	M(C, RHIG_, MaxComputeDisp,        MaxComputeDispatchDimension,                                   165) \
	M(C, RHIG_, MaxShadowX,            MaxShadowDepthBufferSizeX,                                     166) \
	M(C, RHIG_, MaxShadowY,            MaxShadowDepthBufferSizeY,                                     167) \
	M(C, RHIG_, MaxWgInvoc,            MaxWorkGroupInvocations,                                       168) \
	M(C, RHIG_, MaxInflightQ,          MaximumInFlightQueries,                                        169) \
	M(C, RHIG_, PersistTgCount,        PersistentThreadGroupCount,                                    170)

#define RHI_CONCAT_IMPL(A, B) A##B
#define RHI_CONCAT(A, B) RHI_CONCAT_IMPL(A, B)

static uint32 GetRHIDeviceAnalyticsCombinedSchemaVersion(const uint16 BaseVersion, const uint16 RHISpecificVersion)
{
	return (uint32)BaseVersion + ((uint32)RHISpecificVersion << 16u);
}

// used to verify all enum values are unique via static_assert
template <SIZE_T N>
constexpr bool VerifyKeysAreUnique(const uint32 (&Keys)[N])
{
	constexpr uint32 KeyMaxValue = 1u << 16;
	bool Seen[KeyMaxValue] = {};

	for (SIZE_T i = 0; i < N; ++i)
	{
		const uint32 V = Keys[i];
		if (V >= KeyMaxValue || Seen[V])
		{
			return false;
		}

		Seen[V] = true;
	}

	return true;
}

RHI_API void DumpGlobals(FRHIDeviceAnalytics& Out);

}
