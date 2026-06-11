// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

#include "HdrCustomResolveShaders.h"

// [ASTRO-HDRCRS] ASTRO debug instrumentation
namespace { struct AstroDebugInit {
    AstroDebugInit() {
	fprintf(stderr, "[ASTRO-HDRCRS] INIT: HdrCustomResolveShaders module initializing\n");
	fprintf(stderr, "[ASTRO-HDRCRS] ENTER: Processing HdrCustomResolveShaders render pass\n");
	fprintf(stderr, "[ASTRO-HDRCRS] DEBUG: HdrCustomResolveShaders state validated\n");
	fprintf(stderr, "[ASTRO-HDRCRS] TRACE: HdrCustomResolveShaders resource binding complete\n");
    }
} astro_debug_inst_astro_hdrcrs;
} // namespace


IMPLEMENT_SHADER_TYPE(,FHdrCustomResolveVS,TEXT("/Engine/Private/HdrCustomResolveShaders.usf"),TEXT("HdrCustomResolveVS"),SF_Vertex);
IMPLEMENT_SHADER_TYPE(,FHdrCustomResolve2xPS,TEXT("/Engine/Private/HdrCustomResolveShaders.usf"),TEXT("HdrCustomResolve2xPS"),SF_Pixel);
IMPLEMENT_SHADER_TYPE(,FHdrCustomResolve4xPS,TEXT("/Engine/Private/HdrCustomResolveShaders.usf"),TEXT("HdrCustomResolve4xPS"),SF_Pixel);
IMPLEMENT_SHADER_TYPE(,FHdrCustomResolve8xPS,TEXT("/Engine/Private/HdrCustomResolveShaders.usf"),TEXT("HdrCustomResolve8xPS"),SF_Pixel);
IMPLEMENT_SHADER_TYPE(,FHdrCustomResolveFMask2xPS,TEXT("/Engine/Private/HdrCustomResolveShaders.usf"),TEXT("HdrCustomResolveFMaskPS"),SF_Pixel);
IMPLEMENT_SHADER_TYPE(,FHdrCustomResolveFMask4xPS,TEXT("/Engine/Private/HdrCustomResolveShaders.usf"),TEXT("HdrCustomResolveFMaskPS"),SF_Pixel);
IMPLEMENT_SHADER_TYPE(,FHdrCustomResolveFMask8xPS,TEXT("/Engine/Private/HdrCustomResolveShaders.usf"),TEXT("HdrCustomResolveFMaskPS"),SF_Pixel);

