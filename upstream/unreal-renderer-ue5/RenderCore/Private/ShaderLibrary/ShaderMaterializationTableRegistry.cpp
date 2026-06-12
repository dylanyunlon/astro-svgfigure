// Copyright Epic Games, Inc. All Rights Reserved.

#include "ShaderLibrary/ShaderMaterializationTableRegistry.h"
#include "IShaderRuntime.h"
#include "IShaderRuntimeMaterializer.h"
#include "ShaderRuntimeRegistry.h"
#include "ShaderCodeArchive.h"
#include "IO/IoDispatcher.h"
#include "Serialization/MemoryReader.h"
#include "ShaderLibrary/ShaderCodeArchiveInternal.h"

FShaderMaterializationTable::~FShaderMaterializationTable()
{
	if (CanonicalTable)
	{
		UE_LOG(LogShaderLibrary, Display, TEXT("FShaderMaterializationTableRegistry: Unloading canonical table"));
		Materializer->Discard(CanonicalTable);
		CanonicalTable = nullptr;
	}
}

FShaderMaterializationTableRegistry& FShaderMaterializationTableRegistry::Get()
{
	static FShaderMaterializationTableRegistry Instance;
	return Instance;
}

TSharedPtr<FShaderMaterializationTable> FShaderMaterializationTableRegistry::FindOrLoadTable(EShaderPlatform Platform, FIoDispatcher& IoDispatcher)
{
	FScopeLock ScopeLock(&Lock);

	// Check existing tables
	if (TWeakPtr<FShaderMaterializationTable>* Table = Tables.Find(Platform))
	{
		if (TSharedPtr<FShaderMaterializationTable> Pinned = Table->Pin())
		{
			return Pinned;
		}
		
		// Remove dead entry
		Tables.Remove(Platform);
	}

	// Platform may not have a runtime
	IShaderRuntime* Runtime = FShaderRuntimeRegistry::Get().GetRuntime(Platform);
	if (!Runtime)
	{
		return nullptr;
	}

	// Platform may not have a materializer
	IShaderRuntimeMaterializer* Materializer = Runtime->GetMaterializer();
	if (!Materializer)
	{
		return nullptr;
	}

	// No chunk, no table
	FName PlatformName = FDataDrivenShaderPlatformInfo::GetShaderFormat(Platform);
	FIoChunkId MatChunkId = ShaderCodeArchive::GetCanonicalTableChunkId(PlatformName);
	if (!IoDispatcher.DoesChunkExist(MatChunkId))
	{
		UE_LOG(LogShaderLibrary, Display, TEXT("FShaderMaterializationTableRegistry: Canonical table chunk not found for platform %d '%s'"), static_cast<int32>(Platform), *PlatformName.ToString());
		return nullptr;
	}

	// Create request
	FIoBatch   Batch = IoDispatcher.NewBatch();
	FIoRequest Request = Batch.Read(MatChunkId, FIoReadOptions{}, IoDispatcherPriority_Max);

	// Wait for read request
	FEvent* Event = FPlatformProcess::GetSynchEventFromPool();
	Batch.IssueAndTriggerEvent(Event);
	Event->Wait();
	FPlatformProcess::ReturnSynchEventToPool(Event);

	// Any failure is a total failure, there's no way to read shader data without this
	const FIoBuffer& Buffer = Request.GetResultOrDie();
	FMemoryReaderView Ar(MakeArrayView(Buffer.Data(), Buffer.DataSize()));

	// Create table
	TSharedPtr<FShaderMaterializationTable> Table = MakeShared<FShaderMaterializationTable>();
	Table->Materializer = Materializer;
	Table->CanonicalTable = Materializer->CreateCanonicalTable();
	Materializer->Serialize(Table->CanonicalTable, Ar);

	// Keep track
	Tables.Add(Platform, Table);

	UE_LOG(
		LogShaderLibrary, Display, 
		TEXT("FShaderMaterializationTableRegistry: Loaded canonical table for platform %d (%llu bytes)"),
		static_cast<int32>(Platform), Buffer.DataSize()
	);

	return Table;
}
