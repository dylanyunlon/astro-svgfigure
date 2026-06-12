// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "ScenePrimitiveDataEngine.h"

class FScenePrimitiveDataRenderer : public FScenePrimitiveDataEngine
{
public:
	// The data arrays are registered with the manager such that they can share the ID mapping & dirty state tracking.
	FScenePrimitiveDataRenderer();
};

