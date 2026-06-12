// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#if WITH_EDITOR

#include "Tasks/Task.h"
#include "Compression/CompressedBuffer.h"
#include "CoreTypes.h"
#include "Containers/ArrayView.h"
#include "Containers/ContainersFwd.h"
#include "Containers/Map.h"
#include "Containers/Set.h"
#include "HAL/CriticalSection.h"
#include "HAL/Event.h"
#include "Misc/ScopeRWLock.h"
#include "Misc/ScopeLock.h"
#include "Templates/UniquePtr.h"

#include <atomic>

class FZipArchiveWriter;
struct FShaderSymbolFile;

class FShaderSymbolExport
{
public:
	FShaderSymbolExport() = delete;
	RENDERCORE_API FShaderSymbolExport(FName InShaderFormat);
	RENDERCORE_API ~FShaderSymbolExport();

	/** Returns the FShaderSymbolExport for the given shader format, lazily creating on first access. */
	static RENDERCORE_API FShaderSymbolExport& GetOrCreate(FName ShaderFormat);

	/** Drains the export's pending write tasks, finalizes artifacts, and destroys the instance.
	 *  Call once per format at cook shutdown, after all shader compilation callbacks have drained.
	 *  Safe to call on formats that were never created.
	 */
	static RENDERCORE_API void Shutdown(FName ShaderFormat);

	/** Iterates the symbol files and writes each one's decompressed contents to the export. */
	RENDERCORE_API void ExportSymbols(TConstArrayView<FShaderSymbolFile> SymbolFiles, const FString& DebugInfo = FString());

	/** Called at the end of a cook to free resources and finalize artifacts created during the cook. */
	RENDERCORE_API void Finalize();

	/**
	 * Returns true if deterministic symbol export is enabled (driven by the r.Shaders.DeterministicSymbolExport cvar).
	 * When enabled, distinct content variants get distinct filenames via a hash suffix appended on compile job completion;
	 * on shutdown the multiprocess owner coalesces hash-suffixed loose files to the lowest-hash variant per base name.
	 */
	static RENDERCORE_API bool IsDeterministic();

private:
	RENDERCORE_API void Initialize();
	RENDERCORE_API void WriteSymbolData(const FString& Filename, const FString& DebugInfo, TConstArrayView<uint8> Contents);
	void WriteFile(const FString& Filename, const TConstArrayView<uint8>& Contents);
	void CoalesceDeterministicSymbols();

	const FName ShaderFormat;
	const FString ShaderFormatStr;

	TUniquePtr<FZipArchiveWriter> ZipWriter;
	TSet<FString> ExportedShaders;
	FString ExportPath;
	FString InfoFilePath;
	FString ExportFileName;
	uint64 TotalSymbolDataBytes{ 0 };
	uint64 TotalSymbolData{ 0 };
	bool bCanExport{ false };

	TMap<FString, FString> ShaderInfos;

	std::atomic<uint32> DuplicateSymbols{ 0 };

	/**
	 * If true, the current process is the first process in a multiprocess group, or is not in a group,
	 * and should combine artifacts produced by the other processes. Will also be false if no combination
	 * is necessary for given settings.
	 */
	bool bMultiprocessOwner{ false };

	std::atomic<bool> bInitialized{ false };
	FCriticalSection InitCs;

	// Write task synchronization: tasks run concurrently for decompress/deserialize;
	// PendingTaskCount tracks in-flight tasks; AllTasksDoneEvent is triggered when it reaches
	// zero so Finalize can drain without holding references to each task.
	FRWLock SymbolWriteLock;
	std::atomic<int32> PendingTaskCount{ 0 };
	FEventRef AllTasksDoneEvent{ EEventMode::ManualReset };
};

#endif // WITH_EDITOR
