// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/**
 * Wrapper to make it harder to confuse the packed and persistent index when used as arguments etc.
 */
struct FPersistentPrimitiveIndex
{
	bool IsValid() const { return Index != INDEX_NONE; }
	int32 Index = INDEX_NONE;

	FORCEINLINE bool operator == (FPersistentPrimitiveIndex B) const
	{
		return Index == B.Index;
	}

	FORCEINLINE int32 GetIndex() const
	{
		return Index;
	}

	friend uint32 GetTypeHash(FPersistentPrimitiveIndex PersistentPrimitiveIndex)
	{
		return GetTypeHash(PersistentPrimitiveIndex.Index);
	}
};
