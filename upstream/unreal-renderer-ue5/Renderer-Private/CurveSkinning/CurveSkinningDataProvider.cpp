// Copyright Epic Games, Inc. All Rights Reserved.

#include "CurveSkinningDataProvider.h"
#include "ScenePrivate.h"
#include "RenderUtils.h"

IMPLEMENT_SCENE_EXTENSION(FCurveSkinningDataProvider);

bool FCurveSkinningDataProvider::ShouldCreateExtension(FScene& InScene)
{
	return NaniteCurvesSupported() && DoesRuntimeSupportNanite(GetFeatureLevelShaderPlatform(InScene.GetFeatureLevel()), true, true);
}

void FCurveSkinningDataProvider::RegisterProvider(const FCurveSkinningDataProvider::FProviderId& Id, const FOnProvideCurveData& Delegate)
{
#if DO_CHECK
	for (const FCurveDataProvider& ProviderCheck : Providers)
	{
		check(ProviderCheck.Id != Id);
	}
#endif

	check(Delegate.IsBound());
	FCurveDataProvider& Provider = Providers.Emplace_GetRef();
	Provider.Id = Id;
	Provider.Delegate = Delegate;
}

void FCurveSkinningDataProvider::UnregisterProvider(const FCurveSkinningDataProvider::FProviderId& Id)
{
	for (int32 ProviderIndex = 0; ProviderIndex < Providers.Num(); ++ProviderIndex)
	{
		const FCurveDataProvider& Provider = Providers[ProviderIndex];
		if (Provider.Id == Id)
		{
			Providers.RemoveAtSwap(ProviderIndex);
			return;
		}
	}

	checkNoEntry();
}

void FCurveSkinningDataProvider::Broadcast(const TConstArrayView<FProviderRange> Ranges, FProviderContext& Context)
{
	const TConstArrayView<FCurveSkinningDataProvider::FProviderIndirection> IndirectionView = Context.Indirections;
	for (const FCurveDataProvider& Provider : Providers)
	{
		for (const FProviderRange& Range : Ranges)
		{
			if (Provider.Id == Range.Id)
			{
				if (Range.Count > 0)
				{
					Context.Indirections = MakeArrayView(IndirectionView.GetData() + Range.Offset, Range.Count);
					Provider.Delegate.ExecuteIfBound(Context);
				}
				break;
			}
		}
	}
}

bool FCurveSkinningDataProvider::HasProviders() const
{
	return !Providers.IsEmpty();
}

TArray<FCurveSkinningDataProvider::FProviderId> FCurveSkinningDataProvider::GetProviderIds() const
{
	TArray<FProviderId> Ids;
	Ids.Reserve(Providers.Num());
	for (const FCurveDataProvider& Provider : Providers)
	{
		Ids.Add(Provider.Id);
	}
	return Ids;
}