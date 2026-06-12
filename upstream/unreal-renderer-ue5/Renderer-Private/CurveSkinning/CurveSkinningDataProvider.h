// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "SceneExtensions.h"
#include "GameTime.h"

class FCurveSkinningSceneExtensionProxy;

enum class EDirtyCurveData : uint8
{
	None     = 0,
	Current  = 1 << 0,
	Previous = 1 << 1,
	All      = Current | Previous,
};
ENUM_CLASS_FLAGS(EDirtyCurveData);

class FCurveSkinningDataProvider : public ISceneExtension
{
	DECLARE_SCENE_EXTENSION(RENDERER_API, FCurveSkinningDataProvider);

public:
	using FProviderId = FGuid;

	struct FProviderRange
	{
		FProviderId Id;
		uint32 Count;
		uint32 Offset;
	};

	// Per-primitive indirection handed to providers, sliced by FProviderRange.
	struct FProviderIndirection
	{
		uint32          HeaderIndex    = 0;
		uint32          NumCurves      = 0;
		EDirtyCurveData DirtyCurveData = EDirtyCurveData::None;
	};

	// Per-dispatch context handed to each registered curve-data provider.
	//
	// All per-primitive state a provider needs lives in either:
	//   - the curve-skinning header buffer (NumCurves, NumPointsPerCurve, Rest/Deformed offsets, …)
	//   - the GPU scene (per-instance LocalToWorld via Header.InstanceSceneDataOffset)
	//   - the per-instance Indirections array (HeaderIndex + ready-computed byte offsets)
	struct FProviderContext
	{
		FProviderContext(
			const TConstArrayView<FPrimitiveSceneInfo*> InPrimitives,
			const TConstArrayView<FCurveSkinningSceneExtensionProxy*> InProxies,
			const TConstArrayView<FProviderIndirection> InIndirections,
			const FGameTime& InCurrentTime,
			FRDGBuilder& InGraphBuilder,
			FRDGBufferRef InHeaderBuffer,
			FRDGBufferRef InRestBuffer,
			FRDGBufferRef InDeformedBuffer)
			: Primitives(InPrimitives)
			, Proxies(InProxies)
			, Indirections(InIndirections)
			, CurrentTime(InCurrentTime)
			, GraphBuilder(InGraphBuilder)
			, HeaderBuffer(InHeaderBuffer)
			, RestBuffer(InRestBuffer)
			, DeformedBuffer(InDeformedBuffer)
		{
		}

		TConstArrayView<FPrimitiveSceneInfo*>               Primitives;
		TConstArrayView<FCurveSkinningSceneExtensionProxy*> Proxies;
		TConstArrayView<FProviderIndirection>               Indirections;
		const FGameTime&                                    CurrentTime;
		FRDGBuilder&                                        GraphBuilder;

		// Curve-skinning buffers already registered in RDG by FinishCurveSkinningBufferUpload.
		// Providers bind these directly in their compute passes (SRV/SRV/UAV).
		FRDGBufferRef                                       HeaderBuffer;
		FRDGBufferRef                                       RestBuffer;
		FRDGBufferRef                                       DeformedBuffer;
	};

	DECLARE_DELEGATE_OneParam(FOnProvideCurveData, FProviderContext&);

public:
	using ISceneExtension::ISceneExtension;

	static bool ShouldCreateExtension(FScene& InScene);

	RENDERER_API void RegisterProvider(const FProviderId& Id, const FOnProvideCurveData& Delegate);
	RENDERER_API void UnregisterProvider(const FProviderId& Id);

	void Broadcast(const TConstArrayView<FProviderRange> Ranges, FProviderContext& Context);
	bool HasProviders() const;
	TArray<FProviderId> GetProviderIds() const;

private:
	struct FCurveDataProvider
	{
		FProviderId Id;
		FOnProvideCurveData Delegate;
	};

	TArray<FCurveDataProvider> Providers;
};
