// Copyright Epic Games, Inc. All Rights Reserved.

#include "CSMSceneExtension.h"
#include "ScenePrivate.h"
#include "ScenePrimitiveUpdates.h"

IMPLEMENT_SCENE_EXTENSION(FCSMSceneExtension);

FCachedShadowMapData::FCachedShadowMapData(const FWholeSceneProjectedShadowInitializer& InInitializer, float InLastUsedTime)
	: Initializer(InInitializer)
	, LastUsedTime(InLastUsedTime)
	, bCachedShadowMapHasPrimitives(true)
	, bCachedShadowMapHasNaniteGeometry(false)
	, LastFrameExtraStaticShadowSubjects(0)
{
}

void FCSMSceneExtension::InitExtension(FScene& InScene)
{
}

ISceneExtensionUpdater* FCSMSceneExtension::CreateUpdater()
{
	return new FUpdater(*this);
}

void FCSMSceneExtension::RemoveExpiredCacheEntries(float CurrentRealTime)
{
	for (TMap<FLightSceneId, TArray<FCachedShadowMapData>>::TIterator CachedShadowMapIt(CachedShadowMaps); CachedShadowMapIt; ++CachedShadowMapIt)
	{
		TArray<FCachedShadowMapData>& ShadowMapDatas = CachedShadowMapIt.Value();

		for (auto& ShadowMapData : ShadowMapDatas)
		{
			if (ShadowMapData.ShadowMap.IsValid() && CurrentRealTime - ShadowMapData.LastUsedTime > 2.0f)
			{
				ShadowMapData.InvalidateCachedShadow();
			}
		}
	}

	CSV_CUSTOM_STAT_GLOBAL(ShadowCacheUsageMB, (float(GetCachedWholeSceneShadowMapsSize()) / 1024) / 1024, ECsvCustomStatOp::Set);
}

int64 FCSMSceneExtension::GetCachedWholeSceneShadowMapsSize() const
{
	int64 CachedShadowmapMemory = 0;

	for (TMap<FLightSceneId, TArray<FCachedShadowMapData>>::TConstIterator CachedShadowMapIt(CachedShadowMaps); CachedShadowMapIt; ++CachedShadowMapIt)
	{
		const TArray<FCachedShadowMapData>& ShadowMapDatas = CachedShadowMapIt.Value();

		for (const auto& ShadowMapData : ShadowMapDatas)
		{
			if (ShadowMapData.ShadowMap.IsValid())
			{
				CachedShadowmapMemory += ShadowMapData.ShadowMap.ComputeMemorySize();
			}
		}
	}

	return CachedShadowmapMemory;
}

TArray<FCachedShadowMapData>* FCSMSceneExtension::GetCachedShadowMapDatas(FLightSceneId LightId) const
{
	return CachedShadowMaps.Find(LightId);
}

FCachedShadowMapData& FCSMSceneExtension::GetCachedShadowMapDataRef(FLightSceneId LightId, int32 ShadowMapIndex) const
{
	TArray<FCachedShadowMapData>& CachedShadowMapDatas = CachedShadowMaps.FindChecked(LightId);
	checkSlow(ShadowMapIndex >= 0 && ShadowMapIndex < CachedShadowMapDatas.Num());
	return CachedShadowMapDatas[ShadowMapIndex];
}

const FCachedShadowMapData* FCSMSceneExtension::GetCachedShadowMapData(FLightSceneId LightId, int32 ShadowMapIndex) const
{
	const TArray<FCachedShadowMapData>* CachedShadowMapDatas = CachedShadowMaps.Find(LightId);
	if (CachedShadowMapDatas && ShadowMapIndex >= 0 && ShadowMapIndex < CachedShadowMapDatas->Num())
	{
		return &(*CachedShadowMapDatas)[ShadowMapIndex];
	}
	return nullptr;
}

TArray<FCSMDepthReuse>* FCSMSceneExtension::GetCSMDepthReuseData(FLightSceneId LightId) const
{
	return CSMDepthReuseData.Find(LightId);
}

FCSMDepthReuse& FCSMSceneExtension::GetCSMDepthReuseRef(FLightSceneId LightId, int32 ShadowMapIndex) const
{
	TArray<FCSMDepthReuse>& CSMDepthReuseDatas = CSMDepthReuseData.FindChecked(LightId);
	checkSlow(ShadowMapIndex >= 0 && ShadowMapIndex < CSMDepthReuseDatas.Num());
	return CSMDepthReuseDatas[ShadowMapIndex];
}

const FCSMDepthReuse* FCSMSceneExtension::GetCSMDepthReuse(FLightSceneId LightId, int32 ShadowMapIndex) const
{
	const TArray<FCSMDepthReuse>* CSMDepthReuseDatas = CSMDepthReuseData.Find(LightId);
	if (CSMDepthReuseDatas && ShadowMapIndex >= 0 && ShadowMapIndex < CSMDepthReuseDatas->Num())
	{
		return &(*CSMDepthReuseDatas)[ShadowMapIndex];
	}
	return nullptr;
}

int32* FCSMSceneExtension::GetPreviousDynamicCascadeMask(FLightSceneId LightId) const
{
	return PreviousDynamicCascadeMaskPerLight.Find(LightId);
}

void FCSMSceneExtension::SetPreviousDynamicCascadeMask(FLightSceneId LightId, int32 Mask) const
{
	PreviousDynamicCascadeMaskPerLight.FindOrAdd(LightId) = Mask;
}

const FShadowDepthCaptureView* FCSMSceneExtension::GetCapturedView(FLightSceneId LightId) const
{
	if (const FShadowDepthCaptureState* State = CapturedViewStatePerLight.Find(LightId))
	{
		return State->CapturedView.IsSet() ? &State->CapturedView.GetValue() : nullptr;
	}
	return nullptr;
}

bool FCSMSceneExtension::IsShadowDepthFrozen(FLightSceneId LightId) const
{
	if (const FShadowDepthCaptureState* State = CapturedViewStatePerLight.Find(LightId))
	{
		return State->bShadowDepthFrozen;
	}
	return false;
}

void FCSMSceneExtension::SetCapturedView(FLightSceneId LightId, const FShadowDepthCaptureView& View) const
{
	FShadowDepthCaptureState& State = CapturedViewStatePerLight.FindOrAdd(LightId);
	State.CapturedView = View;
	State.bShadowDepthFrozen = false;
}

void FCSMSceneExtension::ClearCapturedView(FLightSceneId LightId) const
{
	CapturedViewStatePerLight.Remove(LightId);
}

void FCSMSceneExtension::NotifyCSMDepthRendered(FLightSceneId LightId) const
{
	if (FShadowDepthCaptureState* State = CapturedViewStatePerLight.Find(LightId))
	{
		if (State->CapturedView.IsSet())
		{
			State->CapturedView.Reset();
			State->bShadowDepthFrozen = true;
		}
	}
}

void FCSMSceneExtension::FUpdater::PreSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePreUpdateChangeSet& ChangeSet)
{
	// Invalidate cached shadows whenever a shadow-casting primitive is removed
	FScene* ScenePtr = static_cast<FScene*>(&CSMScene.Scene);

	for (const FLightSceneInfo* LightSceneInfo : ScenePtr->DirectionalLights)
	{
		TArray<FCachedShadowMapData>* CachedShadowMapDatas = CSMScene.GetCachedShadowMapDatas(LightSceneInfo->Id);

		if (CachedShadowMapDatas)
		{
			for (FCachedShadowMapData& CachedShadowMapData : *CachedShadowMapDatas)
			{
				for (FPersistentPrimitiveIndex PersistentPrimitiveIndex : ChangeSet.RemovedPrimitiveIds)
				{
					if (CachedShadowMapData.StaticShadowSubjectPersistentPrimitiveIdMap[PersistentPrimitiveIndex.Index] == true)
					{
						CachedShadowMapData.InvalidateCachedShadow();
						break;
					}
				}
			}
		}
	}
}

void FCSMSceneExtension::FUpdater::PostSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePostUpdateChangeSet& ChangeSet)
{
	// Resize primitive tracking bitmaps to match new scene size
	FScene* ScenePtr = static_cast<FScene*>(&CSMScene.Scene);
	const int32 MaxPersistentPrimitiveIndex = ScenePtr->GetMaxPersistentPrimitiveIndex();

	for (auto& Pair : CSMScene.CachedShadowMaps)
	{
		for (FCachedShadowMapData& CachedShadowMapData : Pair.Value)
		{
			CachedShadowMapData.StaticShadowSubjectPersistentPrimitiveIdMap.SetNum(MaxPersistentPrimitiveIndex, false);
		}
	}
}

void FCSMSceneExtension::FUpdater::PreLightsUpdate(FRDGBuilder& GraphBuilder, const FLightSceneChangeSet& LightSceneChangeSet)
{
	// Remove cached shadow maps and depth reuse data for removed lights
	for (FLightSceneId LightId : LightSceneChangeSet.RemovedLightIds)
	{
		CSMScene.CachedShadowMaps.Remove(LightId);
		CSMScene.CSMDepthReuseData.Remove(LightId);
		CSMScene.PreviousDynamicCascadeMaskPerLight.Remove(LightId);
		CSMScene.CapturedViewStatePerLight.Remove(LightId);
	}
}
