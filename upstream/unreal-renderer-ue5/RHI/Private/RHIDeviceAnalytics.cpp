// Copyright Epic Games, Inc. All Rights Reserved.

#include "RHIDeviceAnalytics.h"
#include "RHIDeviceAnalyticsInternal.h"
#include "RHIGlobals.h"

namespace RHI::DeviceAnalytics
{
template <typename TValue>
static void EmitGlobal(FRHIDeviceAnalytics& Out, const int32_t Key, const TValue& Value)
{
	if constexpr (
		std::is_same_v<TValue, bool> ||
		std::is_same_v<TValue, int32> ||
		std::is_same_v<TValue, uint32>)
	{
		Out.Add(Key, static_cast<int32>(Value));
	}
	else if constexpr (std::is_same_v<TValue, TRHIGlobal<bool>>)
	{
		Out.Add(Key, static_cast<bool>(Value));
	}
	else if constexpr (std::is_same_v<TValue, TRHIGlobal<int32>>)
	{
		Out.Add(Key, static_cast<int32>(Value));
	}
	else
	{
		static_assert(sizeof(TValue) == 0, "Unknown type");
	}
}

#define DEVICE_ANALYTICS_EMIT(Out, UnusedPrefix, UnusedEnumName, Variable, Num) \
	EmitGlobal(Out, Num, GRHIGlobals.Variable);

RHI_API void DumpGlobals(FRHIDeviceAnalytics& Out)
{
	RHI_DEVICE_ANALYTICS_KEYS_GLOBALS(DEVICE_ANALYTICS_EMIT, Out);
}

#undef DEVICE_ANALYTICS_EMIT

}
