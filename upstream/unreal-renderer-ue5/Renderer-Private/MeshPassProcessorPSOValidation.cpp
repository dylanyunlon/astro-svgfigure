// Copyright Epic Games, Inc. All Rights Reserved.

/*=============================================================================
	MeshPassProcessorPSOValidation.cpp: Renderer-side PSO precache validation helpers.
=============================================================================*/

#include "MeshPassProcessorPSOValidation.h"
#include "MeshPassProcessor.h"
#include "LogRenderer.h"
#include "PrimitiveSceneProxy.h"

#if PSO_PRECACHING_VALIDATE

FGraphicsMinimalPipelineStateInitializer PSOCollectorStats::GetShadersOnlyInitializer(const FGraphicsMinimalPipelineStateInitializer& Initializer)
{
	FGraphicsMinimalPipelineStateInitializer ShadersOnlyInitializer(Initializer);
	ShadersOnlyInitializer.BoundShaderState.VertexDeclarationRHI = nullptr;
	ShadersOnlyInitializer.ComputeStatePrecachePSOHash();

	return ShadersOnlyInitializer;
}

FGraphicsMinimalPipelineStateInitializer PSOCollectorStats::PatchMinimalPipelineStateToCheck(const FGraphicsMinimalPipelineStateInitializer& Initializer)
{
	FGraphicsMinimalPipelineStateInitializer PatchedInitializer = Initializer;
	//PatchedInitializer.DepthStencilState = nullptr;
	//PatchedInitializer.RasterizerState = nullptr;
	//PatchedInitializer.BlendState = nullptr;
	//PatchedInitializer.PrimitiveType = PT_TriangleList;
	//PatchedInitializer.BoundShaderState.VertexDeclarationRHI = nullptr;
	//PatchedInitializer.UniqueEntry = false;

	// Recompute the hash when disabling certain states for checks.
	// PatchedInitializer.StatePrecachePSOHash = 0;
	PatchedInitializer.ComputeStatePrecachePSOHash();

	return PatchedInitializer;
}

uint64 PSOCollectorStats::GetPSOPrecacheHash(const FGraphicsMinimalPipelineStateInitializer& GraphicsPSOInitializer)
{
	return GraphicsPSOInitializer.StatePrecachePSOHash;
}

uint64 PSOCollectorStats::GetShaderPreloadHash(const FMinimalBoundShaderStateInput& BoundShaderState)
{
	TArray<FShaderHash, TInlineAllocator<3>> Hashes;
	if (BoundShaderState.VertexShaderResource && BoundShaderState.VertexShaderIndex != INDEX_NONE)
	{
		Hashes.Add(BoundShaderState.VertexShaderResource->GetShaderHash(BoundShaderState.VertexShaderIndex));
	}
	if (BoundShaderState.PixelShaderResource && BoundShaderState.PixelShaderIndex != INDEX_NONE)
	{
		Hashes.Add(BoundShaderState.PixelShaderResource->GetShaderHash(BoundShaderState.PixelShaderIndex));
	}
#if PLATFORM_SUPPORTS_GEOMETRY_SHADERS
	if (BoundShaderState.GeometryShaderResource && BoundShaderState.GeometryShaderIndex != INDEX_NONE)
	{
		Hashes.Add(BoundShaderState.GeometryShaderResource->GetShaderHash(BoundShaderState.GeometryShaderIndex));
	}
#endif
#if PLATFORM_SUPPORTS_MESH_SHADERS
	if (BoundShaderState.MeshShaderResource && BoundShaderState.MeshShaderIndex != INDEX_NONE)
	{
		Hashes.Add(BoundShaderState.MeshShaderResource->GetShaderHash(BoundShaderState.MeshShaderIndex));
	}
#endif
	return PSOCollectorStats::CombineShaderHashes(Hashes);
}

// Append the "Shader Hashes" lines for a minimal bound shader state. Reads the cooked FShaderHash
// straight off the shader-map resource so we do NOT materialize RHI shaders (AsBoundShaderState would).
static void AppendPSOMissShaderHashes(const FMinimalBoundShaderStateInput& BoundShaderState, PSOMissStringBuilder& StringBuilder)
{
	const auto LogShaderInfo = [&StringBuilder](const TCHAR* ShaderTypeName, FShaderMapResource* Resource, int32 ShaderIndex)
	{
		if (Resource && ShaderIndex != INDEX_NONE)
		{
			StringBuilder.Appendf(TEXT("\n\t\t%s:\t\t%s"), ShaderTypeName, *(Resource->GetShaderHash(ShaderIndex).ToString()));
		}
	};

	LogShaderInfo(TEXT("VertexShader"), BoundShaderState.VertexShaderResource, BoundShaderState.VertexShaderIndex);
	LogShaderInfo(TEXT("PixelShader"), BoundShaderState.PixelShaderResource, BoundShaderState.PixelShaderIndex);
#if PLATFORM_SUPPORTS_GEOMETRY_SHADERS
	LogShaderInfo(TEXT("GeometryShader"), BoundShaderState.GeometryShaderResource, BoundShaderState.GeometryShaderIndex);
#endif
#if PLATFORM_SUPPORTS_MESH_SHADERS
	LogShaderInfo(TEXT("MeshShader"), BoundShaderState.MeshShaderResource, BoundShaderState.MeshShaderIndex);
#endif
}

void PSOCollectorStats::CheckShaderOnlyStateInCache(
	const FGraphicsMinimalPipelineStateInitializer& Initializer,
	const FMaterial& Material,
	const FVertexFactoryType* VFType,
	const FPrimitiveSceneProxy* PrimitiveSceneProxy,
	int32 PSOCollectorIndex)
{
	// The ShadersOnly collector is dual-keyed by precache mode. PSO mode seeds it with the shaders-only
	// state hash; preload mode (producer in AddGraphicsPipelineStateInitializer) seeds it with
	// GetShaderPreloadHash over the cooked shader hashes. The lookup must use the matching recipe.
	uint64 LookupHash = 0;
	if (IsPSOShaderPreloadingEnabled())
	{
		LookupHash = GetShaderPreloadHash(Initializer.BoundShaderState);
	}
	else
	{
		FGraphicsMinimalPipelineStateInitializer ShadersOnlyInitializer = GetShadersOnlyInitializer(Initializer);
		LookupHash = ShadersOnlyInitializer.StatePrecachePSOHash;
	}

	EPSOPrecacheResult Result = EPSOPrecacheResult::Unknown;
	bool bStatsUpdated = PSOCollectorStats::GetShadersOnlyPSOPrecacheStatsCollector().CheckStateInCacheByHash(EPSOPrecacheType::MeshPass, LookupHash, PSOCollectorIndex, VFType, Result);
	if ((IsFullPrecachingValidationEnabled() || IsShaderPreloadingFullValidationEnabled()) && bStatsUpdated && Result != EPSOPrecacheResult::Unknown && Result != EPSOPrecacheResult::Complete)
	{
		LogPSOMissInfoShadersOnly(Initializer, Result, &Material, VFType, PrimitiveSceneProxy, EPSOPrecacheType::MeshPass, PSOCollectorIndex, LookupHash);
	}
}

void PSOCollectorStats::CheckMinimalPipelineStateInCache(
	const FGraphicsMinimalPipelineStateInitializer& Initializer,
	const FMaterial& Material,
	const FVertexFactoryType* VFType,
	const FPrimitiveSceneProxy* PrimitiveSceneProxy,
	int32 PSOCollectorIndex)
{
	FGraphicsMinimalPipelineStateInitializer PatchedMinimalInitializer = PSOCollectorStats::PatchMinimalPipelineStateToCheck(Initializer);

	EPSOPrecacheResult Result = EPSOPrecacheResult::Unknown;
	bool bStatsUpdated = PSOCollectorStats::GetMinimalPSOPrecacheStatsCollector().CheckStateInCacheByHash(EPSOPrecacheType::MeshPass, PatchedMinimalInitializer.StatePrecachePSOHash, PSOCollectorIndex, VFType, Result);
	if (IsFullPrecachingValidationEnabled() && bStatsUpdated && Result != EPSOPrecacheResult::Unknown && Result != EPSOPrecacheResult::Complete)
	{
		FGraphicsMinimalPipelineStateInitializer ShadersOnlyInitializer = GetShadersOnlyInitializer(Initializer);
		bool bShaderOnlyPrecached = PSOCollectorStats::GetShadersOnlyPSOPrecacheStatsCollector().IsPrecached(ShadersOnlyInitializer.StatePrecachePSOHash);
		if (bShaderOnlyPrecached)
		{
			check(Result == EPSOPrecacheResult::Missed);
			FGraphicsPipelineStateInitializer GraphicsPSOInitializer = PatchedMinimalInitializer.AsGraphicsPipelineStateInitializer();
			LogPSOMissInfo(GraphicsPSOInitializer, EPSOPrecacheMissType::MinimalPSOState, Result, &Material, VFType, PrimitiveSceneProxy, EPSOPrecacheType::MeshPass, PSOCollectorIndex, ShadersOnlyInitializer.StatePrecachePSOHash);
		}
	}
}

void LogPSOMissInfoShadersOnly(
	const FGraphicsMinimalPipelineStateInitializer& GraphicsPSOInitializer,
	EPSOPrecacheResult PrecacheResult,
	const FMaterial* Material,
	const FVertexFactoryType* VFType,
	const FPrimitiveSceneProxy* PrimitiveSceneProxy,
	EPSOPrecacheType PSOPrecacheType,
	int32 PSOCollectorIndex,
	uint64 ShadersOnlyPSOInitializerHash)
{
	PSOMissStringBuilder StringBuilder;
	LogGeneralPSOMissInfo(Material, VFType, PrimitiveSceneProxy, PSOPrecacheType, PSOCollectorIndex, EPSOPrecacheMissType::ShadersOnly, PrecacheResult, StringBuilder);
	AppendPSOMissShaderHashes(GraphicsPSOInitializer.BoundShaderState, StringBuilder);

#if PSO_PRECACHING_TRACKING
	if (PrecacheResult == EPSOPrecacheResult::Untracked)
	{
		StringBuilder << TEXT("\n\n\tUntracked Info:");
		if (!VFType->SupportsPSOPrecaching())
		{
			StringBuilder << TEXT("\n\t\t- VertexFactory doesn't support PSO precaching.");
		}
		if (!PSOCollectorStats::FPrecacheStatsCollector::IsStateTracked(PSOPrecacheType, PSOCollectorIndex, nullptr))
		{
			StringBuilder << TEXT("\n\t\t- MeshPassProcessor doesn't support PSO precaching.");
		}
	}
	else if (PrecacheResult == EPSOPrecacheResult::Missed)
	{
		check(Material);
		StringBuilder << TEXT("\n\n\tMissed Info:");
		LogMaterialPSOPrecacheRequestData(*Material, VFType, StringBuilder);
	}
#endif // PSO_PRECACHING_TRACKING

	UE_LOGF(LogRenderer, Log, "%ls\n", StringBuilder.ToString());
}

#endif // PSO_PRECACHING_VALIDATE
