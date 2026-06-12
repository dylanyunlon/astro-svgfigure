// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"

class FRHICommandListImmediate;

class ICustomResourcePool
{
public:
	virtual ~ICustomResourcePool() = default;
	virtual void Tick(FRHICommandListImmediate& RHICmdList) = 0;

	static RENDERCORE_API void TickPoolElements(FRHICommandListImmediate& RHICmdList);
};
extern RENDERCORE_API ICustomResourcePool* GCustomResourcePool;
