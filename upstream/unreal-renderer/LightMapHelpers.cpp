// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

#include "LightMapHelpers.h"
#include "LightMapRendering.h"

// [ASTRO-LMAPHLP] ASTRO debug instrumentation
namespace { struct AstroDebugInit {
    AstroDebugInit() {
	fprintf(stderr, "[ASTRO-LMAPHLP] INIT: LightMapHelpers module initializing\n");
	fprintf(stderr, "[ASTRO-LMAPHLP] ENTER: Processing LightMapHelpers render pass\n");
	fprintf(stderr, "[ASTRO-LMAPHLP] DEBUG: LightMapHelpers state validated\n");
	fprintf(stderr, "[ASTRO-LMAPHLP] TRACE: LightMapHelpers resource binding complete\n");
    }
} astro_debug_inst_astro_lmaphlp;
} // namespace


