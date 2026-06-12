// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "HAL/Platform.h"
#include "Misc/TVariant.h"

// RHI device analytics report.
// Contains list of key value pairs describing the RHI: features, limits, properties, extensions, formats, etc.
// Key can be either FString for values that are discovered in runtime, or an int32 for keys known at compile time to reduce binary size.
// Int32 keys can be mapped back to strings via RHI specific schema. 
struct FRHIDeviceAnalytics
{
	using FKey   = TVariant<int32, FString>;
	using FValue = TVariant<int32, int64, float, FString>;

	struct FKeyValue
	{
		FKey   Key;
		FValue Value;
	};

	// defines mapping of numerical keys
	uint32 SchemaVersion = 0;

	// all key-value pairs
	TArray<FKeyValue> Pairs;

	template <typename TKey, typename TValue>
	void Add(TKey&& InKey, TValue&& InValue)
	{
		FKeyValue& E = Pairs.AddDefaulted_GetRef();

		using KeyT = std::decay_t<TKey>;
		if constexpr (std::is_enum_v<KeyT> || std::is_same_v<KeyT, int32>)
		{
			E.Key.Set<int32>(static_cast<int32>(InKey));
		}
		else
		{
			E.Key.Set<FString>(FString(InKey));
		}

		using ValueT = std::decay_t<TValue>;
		if constexpr (std::is_same_v<ValueT, bool>)
		{
			E.Value.Set<int32>(InValue ? 1 : 0);
		}
		else if constexpr (std::is_same_v<ValueT, int32>)
		{
			E.Value.Set<int32>(InValue);
		}
		else if constexpr (std::is_same_v<ValueT, int64>)
		{
			E.Value.Set<int64>(InValue);
		}
		else if constexpr (std::is_same_v<ValueT, float>)
		{
			E.Value.Set<float>(InValue);
		}
		else
		{
			E.Value.Set<FString>(FString(Forward<TValue>(InValue)));
		}
	}
};
