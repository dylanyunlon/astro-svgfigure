// Copyright Epic Games, Inc. All Rights Reserved.

#include "RHILockTracker.h"
#include "RHI.h"

FRHILockTracker GRHILockTracker;

void FRHILockTracker::RaiseMismatchError()
{
	UE_LOGF(LogRHI, Fatal, "Mismatched RHI buffer locks.");
}
