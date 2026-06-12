// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "RHIShaderPlatform.h"

class IShaderRuntimeMaterializer;
struct FShaderRuntimeCanonicalTable;
class FIoDispatcher;

struct FShaderMaterializationTable
{
	RENDERCORE_API ~FShaderMaterializationTable();
	
	/** Owning materializer */
	IShaderRuntimeMaterializer* Materializer = nullptr;
	
	/** Canonical table */
	FShaderRuntimeCanonicalTable* CanonicalTable = nullptr;
};

class FShaderMaterializationTableRegistry
{
public:
	/** Get the instance */
	RENDERCORE_API static FShaderMaterializationTableRegistry& Get();

	/** Find or load a table */
	RENDERCORE_API TSharedPtr<FShaderMaterializationTable> FindOrLoadTable(EShaderPlatform Platform, FIoDispatcher& IoDispatcher);

private:
	/** Shared lock */
	FCriticalSection Lock;
	
	/** All existing tables */
	TMap<EShaderPlatform, TWeakPtr<FShaderMaterializationTable>> Tables;
};
