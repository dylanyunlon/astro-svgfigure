// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreTypes.h"
#include "Misc/Optional.h"

enum class ERHIThreadMode
{
	None,
	DedicatedThread,
	Tasks
};

// Start the dedicated RHI thread
void StartRHIThread();

// Stop the dedicated RHI thread
void StopRHIThread();

extern TOptional<ERHIThreadMode> GPendingRHIThreadMode;
