// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "ShadowDepthCaptureView.h"
#include "SceneExtensions.h"
#include "SceneManagement.h"
#include "SceneRendering.h"

class FScene;
struct FLightSceneChangeSet;
class FScenePreUpdateChangeSet;
class FScenePostUpdateChangeSet;
class FSceneUniformBuffer;

// CSM shadow depth reuse CVars (defined in ShadowSetup.cpp)
extern TAutoConsoleVariable<bool> CVarCSMShadowDepthReuse;
extern TAutoConsoleVariable<int32> CVarCSMShadowDepthReuseSeparateDynamic;
extern TAutoConsoleVariable<int32> CVarCSMShadowDepthReuseDynamicCascadeMask;
extern TAutoConsoleVariable<int32> CVarCSMShadowDepthReuseInterval;
extern TAutoConsoleVariable<int32> CVarCSMShadowDepthReuseDynamicInterval;
extern TAutoConsoleVariable<int32> CVarCSMShadowDepthReuseFarInterval;

/**
 * Cached shadow map data for whole-scene shadows (CSM, point, spot).
 */
class FCachedShadowMapData
{
public:
	FWholeSceneProjectedShadowInitializer Initializer;
	FShadowMapRenderTargetsRefCounted ShadowMap;
	float LastUsedTime;
	bool bCachedShadowMapHasPrimitives;
	bool bCachedShadowMapHasNaniteGeometry;

	/**
	 * The static meshes cast shadow on this cached csm
	 */
	TBitArray<> StaticShadowSubjectPersistentPrimitiveIdMap;

	FIntPoint ShadowBufferResolution;
	FVector PreShadowTranslation;
	float MaxSubjectZ;
	float MinSubjectZ;

	/**
	 * The extra static meshes cast shadow in last frame, if it exceeds the r.Shadow.MaxCSMScrollingStaticShadowSubjects, the cached csm should be rebuilt.
	 */
	int32 LastFrameExtraStaticShadowSubjects;

	void InvalidateCachedShadow()
	{
		ShadowMap.Release();
		StaticShadowSubjectPersistentPrimitiveIdMap.SetRange(0, StaticShadowSubjectPersistentPrimitiveIdMap.Num(), false);
	}

	FCachedShadowMapData(const FWholeSceneProjectedShadowInitializer& InInitializer, float InLastUsedTime);
};

/**
 * Captured-view freeze state for the CSM shadow-depth capture flow. One entry per directional
 * light that has an active capture (or has had one and is now in the frozen reuse phase).
 *
 * Lifecycle:
 *   SetCapturedView    : CapturedView.IsSet() == true,  bShadowDepthFrozen = false  (capture queued)
 *   NotifyCSMDepthRendered (called by ShadowDepthRendering after a successful CSM atlas write):
 *                        CapturedView.Reset(),          bShadowDepthFrozen = true   (now frozen)
 *   ClearCapturedView  : entry removed
 */
class FShadowDepthCaptureState
{
public:
	TOptional<FShadowDepthCaptureView> CapturedView;
	bool bShadowDepthFrozen = false;
};

/**
 * Per-cascade depth reuse data for CSM shadow depth amortization.
 */
class FCSMDepthReuse
{
public:
	bool bHasEverRendered;
	uint32 PreviousRenderDepthFrameNumber;
	FShadowMapRenderTargetsRefCounted PreviousAtlas;
	FMatrix44f TranslatedWorldToClipInnerMatrix;
	FMatrix44f TranslatedWorldToClipOuterMatrix;
	float MaxSubjectZ;
	float MinSubjectZ;
	FVector PreShadowTranslation;
	uint32 X;
	uint32 Y;
	float FrozenSplitFar;
	float FrozenSplitNear;
	float FrozenFadePlaneOffset;
	float FrozenFadePlaneLength;
	FVector FrozenViewOrigin;
	FVector FrozenViewForward;

	FCSMDepthReuse()
		: bHasEverRendered(false)
		, PreviousRenderDepthFrameNumber(0)
		, X(0)
		, Y(0)
		, FrozenSplitFar(0.0f)
		, FrozenSplitNear(0.0f)
		, FrozenFadePlaneOffset(0.0f)
		, FrozenFadePlaneLength(0.0f)
		, FrozenViewOrigin(FVector::ZeroVector)
		, FrozenViewForward(FVector::ZeroVector)
	{}
};

/**
 * Scene extension for Cascaded Shadow Map (CSM) state management.
 * This extension owns all persistent CSM-related state.
 */
class FCSMSceneExtension : public ISceneExtension
{
	DECLARE_SCENE_EXTENSION(RENDERER_API, FCSMSceneExtension);

public:
	using ISceneExtension::ISceneExtension;

	//~ ISceneExtension interface
	virtual void InitExtension(FScene& InScene) override;
	virtual ISceneExtensionUpdater* CreateUpdater() override;

	/**
	 * Remove cached shadow map entries that haven't been used recently and report stats.
	 */
	void RemoveExpiredCacheEntries(float CurrentRealTime);

	/**
	 * Get the total memory used by all cached shadow maps.
	 */
	int64 GetCachedWholeSceneShadowMapsSize() const;

	/**
	 * Cached shadow map accessors. All const-qualified since the underlying data is mutable cache state.
	 */
	TArray<FCachedShadowMapData>* GetCachedShadowMapDatas(FLightSceneId LightId) const;
	FCachedShadowMapData& GetCachedShadowMapDataRef(FLightSceneId LightId, int32 ShadowMapIndex = 0) const;
	const FCachedShadowMapData* GetCachedShadowMapData(FLightSceneId LightId, int32 ShadowMapIndex = 0) const;
	TMap<FLightSceneId, TArray<FCachedShadowMapData>>& GetAllCachedShadowMaps() const { return CachedShadowMaps; }

	/**
	 * CSM depth reuse accessors. All const-qualified since the underlying data is mutable cache state.
	 */
	TArray<FCSMDepthReuse>* GetCSMDepthReuseData(FLightSceneId LightId) const;
	FCSMDepthReuse& GetCSMDepthReuseRef(FLightSceneId LightId, int32 ShadowMapIndex = 0) const;
	const FCSMDepthReuse* GetCSMDepthReuse(FLightSceneId LightId, int32 ShadowMapIndex = 0) const;
	TMap<FLightSceneId, TArray<FCSMDepthReuse>>& GetAllCSMDepthReuseData() const { return CSMDepthReuseData; }

	/**
	 * Per-light tracking of previous dynamic cascade mask for invalidation on change.
	 */
	int32* GetPreviousDynamicCascadeMask(FLightSceneId LightId) const;
	void SetPreviousDynamicCascadeMask(FLightSceneId LightId, int32 Mask) const;

	const FShadowDepthCaptureView* GetCapturedView(FLightSceneId LightId) const;
	bool IsShadowDepthFrozen(FLightSceneId LightId) const;
	void SetCapturedView(FLightSceneId LightId, const FShadowDepthCaptureView& View) const;
	void ClearCapturedView(FLightSceneId LightId) const;
	void NotifyCSMDepthRendered(FLightSceneId LightId) const;

private:
	class FUpdater : public ISceneExtensionUpdater
	{
	public:
		DECLARE_SCENE_EXTENSION_UPDATER(FUpdater, FCSMSceneExtension);

		FUpdater(FCSMSceneExtension& InCSMScene) : CSMScene(InCSMScene) {}

		virtual void PreSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePreUpdateChangeSet& ChangeSet) override;
		virtual void PostSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePostUpdateChangeSet& ChangeSet) override;
		virtual void PreLightsUpdate(FRDGBuilder& GraphBuilder, const FLightSceneChangeSet& LightSceneChangeSet) override;

		FCSMSceneExtension& CSMScene;
	};

	friend class FUpdater;

	/** Map from light id to the cached shadowmap data for that light. Mutable as this is cache state modified during rendering. */
	mutable TMap<FLightSceneId, TArray<FCachedShadowMapData>> CachedShadowMaps;

	/** Map from light id to per-cascade depth reuse data. Mutable as this is cache state modified during rendering. */
	mutable TMap<FLightSceneId, TArray<FCSMDepthReuse>> CSMDepthReuseData;

	/** Map from light id to the previously applied dynamic cascade mask, for invalidation on change. */
	mutable TMap<FLightSceneId, int32> PreviousDynamicCascadeMaskPerLight;

	/** Mutable: render-thread accessors mutate cache state through const member functions. */
	mutable TMap<FLightSceneId, FShadowDepthCaptureState> CapturedViewStatePerLight;
};
