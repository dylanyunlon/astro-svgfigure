// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreTypes.h"

struct FSuspendRenderingTickables
{
	FSuspendRenderingTickables();
	~FSuspendRenderingTickables();
};

void StartRenderingThreadHeartbeatThread(uint32 ThreadNumber);
void StopRenderingThreadHeartbeatThread();
