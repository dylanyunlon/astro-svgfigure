// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

enum class EDiffuseIndirectMethod
{
	Disabled,
	SSGI,
	Lumen,
	Plugin,
};

enum class EAmbientOcclusionMethod
{
	Disabled,
	SSAO,
	SSGI, // SSGI can produce AO buffer at same time to correctly comp SSGI within the other indirect light such as skylight and lightmass.
	RTAO,
};

enum class EReflectionsMethod
{
	Disabled,
	SSR,
	Lumen
};
