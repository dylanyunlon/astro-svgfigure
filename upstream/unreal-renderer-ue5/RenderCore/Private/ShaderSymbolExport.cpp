// Copyright Epic Games, Inc. All Rights Reserved.

#include "ShaderSymbolExport.h"

#if WITH_EDITOR

#include "FileUtilities/ZipArchiveReader.h"
#include "FileUtilities/ZipArchiveWriter.h"
#include "GenericPlatform/GenericPlatformFile.h"
#include "HAL/FileManager.h"
#include "HAL/IConsoleManager.h"
#include "HAL/PlatformFileManager.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTLS.h"
#include "Logging/LogMacros.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Misc/PathViews.h"
#include "ShaderCore.h"
#include "ShaderCompilerCore.h"

DECLARE_LOG_CATEGORY_CLASS(LogShaderSymbolExport, Display, Display);

static TAutoConsoleVariable<bool> CVarShadersDeterministicSymbolExport(
	TEXT("r.Shaders.DeterministicSymbolExport"),
	false,
	TEXT("When enabled, shader symbol export writes all symbol variants with content hashes in filenames,\n")
	TEXT("then coalesces to the single lowest-hash version on shutdown to ensure deterministic output.\n")
	TEXT("Only affects loose file export mode. Note that this incurs additional I/O cost."),
	ECVF_ReadOnly);

static const TCHAR* ZipFileExtension = TEXT(".zip");
static const TCHAR* InfoFileExtension = TEXT(".info");

FShaderSymbolExport::FShaderSymbolExport(FName InShaderFormat)
	: ShaderFormat(InShaderFormat)
	, ShaderFormatStr(InShaderFormat.ToString())
{
}

FShaderSymbolExport::~FShaderSymbolExport() = default;

namespace UE::Shaders::Private
{
	static FRWLock GSymbolExportLock;
	static TMap<FName, TUniquePtr<FShaderSymbolExport>> GSymbolExports;
}

FShaderSymbolExport& FShaderSymbolExport::GetOrCreate(FName ShaderFormat)
{
	using namespace UE::Shaders::Private;

	// Fast path: read-lock lookup for the common case where the export already exists.
	{
		FRWScopeLock ReadLock(GSymbolExportLock, SLT_ReadOnly);
		if (TUniquePtr<FShaderSymbolExport>* Entry = GSymbolExports.Find(ShaderFormat))
		{
			return **Entry;
		}
	}

	// Slow path: promote to write-lock and re-check before creating.
	FRWScopeLock WriteLock(GSymbolExportLock, SLT_Write);
	TUniquePtr<FShaderSymbolExport>& Entry = GSymbolExports.FindOrAdd(ShaderFormat);
	if (!Entry)
	{
		Entry = MakeUnique<FShaderSymbolExport>(ShaderFormat);
	}
	return *Entry;
}

bool FShaderSymbolExport::IsDeterministic()
{
	return CVarShadersDeterministicSymbolExport.GetValueOnAnyThread();
}

void FShaderSymbolExport::Shutdown(FName ShaderFormat)
{
	using namespace UE::Shaders::Private;

	TUniquePtr<FShaderSymbolExport> Export;
	{
		FRWScopeLock WriteLock(GSymbolExportLock, SLT_Write);
		GSymbolExports.RemoveAndCopyValue(ShaderFormat, Export);
	}
	if (Export)
	{
		Export->Finalize();
	}
}

static void DeleteExisting(IPlatformFile& PlatformFile, const FString& Directory, const FString& BaseLeafName, const TCHAR* Extension)
{
	TArray<FString> ExistingZips;
	PlatformFile.FindFiles(ExistingZips, *Directory, Extension);

	// To minimize over-deleting match the full name [BaseLeafName.]ext or the workers [BaseLeafName_]N.ext
	FString BaseLeafRoot = BaseLeafName + TEXT(".");
	FString BaseLeafWorker = BaseLeafName + TEXT("_");

	for (const FString& ZipFile : ExistingZips)
	{
		const FStringView Leaf = FPathViews::GetPathLeaf(ZipFile);
		if (Leaf.StartsWith(BaseLeafRoot) || Leaf.StartsWith(BaseLeafWorker))
		{
			PlatformFile.DeleteFile(*ZipFile);
		}
	}
}

static FString CreateNameAndDeleteOld(uint32 MultiprocessId, IPlatformFile& PlatformFile, const FString& ExportPath, const FString& BaseLeafName, const TCHAR* Extension)
{
	FString Name;
	if (MultiprocessId == 0)
	{
		DeleteExisting(PlatformFile, ExportPath, BaseLeafName, Extension);
		Name = FString::Printf(TEXT("%s%s"), *BaseLeafName, Extension);
	}
	else
	{
		Name = FString::Printf(TEXT("%s_%d%s"), *BaseLeafName, MultiprocessId, Extension);
	}
	return Name;
}

void FShaderSymbolExport::Initialize()
{
	const bool bSymbolExportEnabled = ShouldWriteShaderSymbols(ShaderFormat);
	const bool bSymbolGenerationEnabled = ShouldGenerateShaderSymbols(ShaderFormat);
	const bool bForceSymbols = FParse::Value(FCommandLine::Get(), TEXT("-ShaderSymbolsExport="), ExportPath);
	const bool bSymbolsInfoEnabled = ShouldGenerateShaderSymbolsInfo(ShaderFormat);

	if (bSymbolExportEnabled || bForceSymbols || bSymbolsInfoEnabled)
	{
		// if no command line path is provided, look to the cvar first
		if (ExportPath.IsEmpty())
		{
			if (GetShaderSymbolPathOverride(ExportPath, ShaderFormat))
			{
				ExportPath = IFileManager::Get().ConvertToAbsolutePathForExternalAppForWrite(*ExportPath);
			}
		}

		// if there was no path set via command line or the cvar, fall back to our default
		if (ExportPath.IsEmpty())
		{
			ExportPath = IFileManager::Get().ConvertToAbsolutePathForExternalAppForWrite(
				*(FPaths::ProjectSavedDir() / TEXT("ShaderSymbols") / ShaderFormatStr));
		}

		// Setup ExportFileName to the default or the cvar override
		ExportFileName = TEXT("ShaderSymbols");
		GetShaderFileNameOverride(ExportFileName, TEXT("r.Shaders.SymbolFileNameOverride"), ShaderFormat, ShaderFormat);

		IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
		bCanExport = PlatformFile.CreateDirectoryTree(*ExportPath);

		if (!bCanExport)
		{
			UE_LOGF(LogShaderSymbolExport, Error, "Failed to create shader symbols output directory. Shader symbol export will be disabled.");
		}
		else
		{
			// setup multiproc data in case we need it
			uint32 MultiprocessId = UE::GetMultiprocessId();
			bMultiprocessOwner = MultiprocessId == 0;

			// Check if the export mode is to an uncompressed/compressed archive or loose files.
			const EWriteShaderSymbols WriteShaderSymbolsOptions = GetWriteShaderSymbolsOptions(ShaderFormat);
			const bool bExportAsZip = WriteShaderSymbolsOptions != EWriteShaderSymbols::Disable;

			if (bSymbolExportEnabled && (bExportAsZip || FParse::Param(FCommandLine::Get(), TEXT("ShaderSymbolsExportZip"))))
			{
				FString LeafName = CreateNameAndDeleteOld(MultiprocessId, PlatformFile, ExportPath, ExportFileName, ZipFileExtension);
				FString SingleFilePath = ExportPath / LeafName;

				IFileHandle* OutputZipFile = PlatformFile.OpenWrite(*SingleFilePath);
				if (!OutputZipFile)
				{
					UE_LOGF(LogShaderSymbolExport, Error, "Failed to create shader symbols output file \"%ls\". Shader symbol export will be disabled.", *SingleFilePath);
					bCanExport = false;
				}
				else
				{
					// only compress the first zip file. Other ones are going to be compress in the merging
					EZipArchiveOptions ZipOptions = (WriteShaderSymbolsOptions == EWriteShaderSymbols::Compress && bMultiprocessOwner) ? EZipArchiveOptions::Deflate : EZipArchiveOptions::None;
					ZipOptions |= EZipArchiveOptions::RemoveDuplicate;
					ZipWriter = MakeUnique<FZipArchiveWriter>(OutputZipFile, ZipOptions);
				}
			}
			
			if (bSymbolsInfoEnabled)
			{
				// if we are exporting collated shader pdb info into one file
				FString LeafName = CreateNameAndDeleteOld(MultiprocessId, PlatformFile, ExportPath, ExportFileName, InfoFileExtension);
				InfoFilePath = ExportPath / LeafName;
			}
		}
	}

	if (bSymbolGenerationEnabled || (bCanExport && (bSymbolExportEnabled || bSymbolsInfoEnabled)))
	{
		UE_LOGF(LogShaderSymbolExport, Display, "Shader symbol export settings for format %ls: symbols info %ls, symbol generation %ls, symbol export %ls%ls. Output directory: \"%ls\"",
			*ShaderFormatStr,
			bSymbolsInfoEnabled ? TEXT("ON") : TEXT("OFF"),
			bSymbolGenerationEnabled ? TEXT("ON") : TEXT("OFF"),
			bSymbolExportEnabled ? TEXT("ON") : TEXT("OFF"),
			IsDeterministic() ? TEXT(" (deterministic)") : TEXT(""), 
			*ExportPath);
			
		if (ZipWriter)
		{
			UE_LOGF(LogShaderSymbolExport, Display, "Shader symbol zip mode enabled. Shader symbols will be archived in a single (uncompressed) zip file.");
		}
	}
}

void FShaderSymbolExport::WriteSymbolData(const FString& Filename, const FString& DebugData, TConstArrayView<uint8> Contents)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(FShaderSymbolExport::WriteSymbolData);

	if (Filename.IsEmpty())
	{
		return;
	}

	const bool bShouldWriteSymbols = ShouldWriteShaderSymbols(ShaderFormat) && Contents.Num();
	const bool bShouldWriteToZip = ZipWriter != nullptr;
	const bool bWriteLooseFile = bShouldWriteSymbols && !bShouldWriteToZip;

	// Fast read-only check: is this shader already tracked?
	bool bAlreadySeen = false;
	{
		FRWScopeLock ReadLock(SymbolWriteLock, SLT_ReadOnly);
		bAlreadySeen = ExportedShaders.Contains(Filename);
	}

	if (bAlreadySeen)
	{
		DuplicateSymbols.fetch_add(1, std::memory_order_relaxed);
	}
	else
	{
		// First encounter (tentative): acquire write lock and re-check for races.
		FRWScopeLock WriteLock(SymbolWriteLock, SLT_Write);

		bool bAlreadyInSet = false;
		ExportedShaders.Add(Filename, &bAlreadyInSet);

		if (bAlreadyInSet)
		{
			bAlreadySeen = true;
			DuplicateSymbols.fetch_add(1, std::memory_order_relaxed);
		}
		else
		{
			static uint64 LastReport = 0;
			TotalSymbolDataBytes += Contents.Num();
			TotalSymbolData++;

			if ((TotalSymbolDataBytes - LastReport) >= (64 * 1024 * 1024))
			{
				UE_LOGF(LogShaderSymbolExport, Display, "%ls shader symbols export size: %.2f MB, count: %llu",
					*ShaderFormatStr,
					double(TotalSymbolDataBytes) / (1024.0 * 1024.0), TotalSymbolData);
				LastReport = TotalSymbolDataBytes;
			}

			if (ShouldGenerateShaderSymbolsInfo(ShaderFormat) && !DebugData.IsEmpty())
			{
				FString FilenameNoExt = FPaths::GetBaseFilename(Filename);
				ShaderInfos.Add({ FilenameNoExt, DebugData });
			}

			if (bShouldWriteSymbols && bShouldWriteToZip)
			{
				ZipWriter->AddFile(Filename, Contents, FDateTime::Now());
			}
		}
	}

	// Loose file write (outside lock scope). When deterministic export is on, the hash suffix
	// is already baked into Filename upstream by compile job processing, so distinct variants
	// produce distinct names and bAlreadySeen stays false for each — no extra branching needed.
	if (bWriteLooseFile && !bAlreadySeen)
	{
		WriteFile(Filename, Contents);
	}
}

void FShaderSymbolExport::WriteFile(const FString& Filename, const TConstArrayView<uint8>& Contents)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(FShaderSymbolExport::WriteFile);
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

	const FString OutputPath = ExportPath / Filename;
	const FString Directory = FPaths::GetPath(OutputPath);

	// Filename could contain extra folders, so we need to make sure they exist first.
	if (!PlatformFile.CreateDirectoryTree(*Directory))
	{
		UE_LOGF(LogShaderSymbolExport, Error, "Failed to create shader symbol directory \"%ls\".", *Directory);
		return;
	}

	// Write to a unique temp file (pid_tid suffix), then rename to target.
	// Handles concurrent writes from multiple processes: if another process
	// already wrote the target, the rename fails and we clean up silently.
	const FString TempPath = FString::Printf(TEXT("%s.tmp_%u_%u"), *OutputPath,
		FPlatformProcess::GetCurrentProcessId(), FPlatformTLS::GetCurrentThreadId());

	{
		TUniquePtr<IFileHandle> File(PlatformFile.OpenWrite(*TempPath));
		if (!File)
		{
			UE_LOGF(LogShaderSymbolExport, Error, "Failed to create temp shader symbols file \"%ls\".", *TempPath);
			return;
		}
		if (!File->Write(Contents.GetData(), Contents.Num()))
		{
			UE_LOGF(LogShaderSymbolExport, Error, "Failed to write shader symbols \"%ls\".", *TempPath);
			File.Reset();
			PlatformFile.DeleteFile(*TempPath);
			return;
		}
	}

	if (!PlatformFile.MoveFile(*OutputPath, *TempPath))
	{
		// Another process already wrote this file; clean up our temp copy.
		PlatformFile.DeleteFile(*TempPath);
	}
}

void FShaderSymbolExport::CoalesceDeterministicSymbols()
{
	TRACE_CPUPROFILER_EVENT_SCOPE(FShaderSymbolExport::CoalesceDeterministicSymbols);
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

	// Single-pass scan: for each hash-suffixed file, keep only the lowest-hash variant per
	// base path. Higher-hash files and superseded files are deleted immediately during iteration.
	struct FBestFile
	{
		uint64 Hash;
		FString FullPath;
	};
	TMap<FString, FBestFile> BestPerBasePath;

	PlatformFile.IterateDirectoryRecursively(*ExportPath,
		[&BestPerBasePath, &PlatformFile](const TCHAR* FilenameOrDirectory, bool bIsDirectory) -> bool
		{
			if (bIsDirectory)
			{
				return true;
			}

			// Hash-suffixed files have the form: OriginalName.ext.XXXXXXXXXXXXXXXX
			// where the appended extension is exactly 16 hex characters.
			FStringView FullPath(FilenameOrDirectory);
			FStringView Ext = FPathViews::GetExtension(FullPath);
			if (Ext.Len() != 16)
			{
				return true;
			}

			for (TCHAR Ch : Ext)
			{
				if (!FChar::IsHexDigit(Ch))
				{
					return true;
				}
			}

			uint64 Hash = FParse::HexNumber64(Ext);
			FString BasePath(FullPath.Left(FullPath.Len() - 17)); // strip ".XXXXXXXXXXXXXXXX"

			FBestFile* Existing = BestPerBasePath.Find(BasePath);
			if (!Existing)
			{
				BestPerBasePath.Add(MoveTemp(BasePath), { Hash, FString(FullPath) });
			}
			else if (Hash < Existing->Hash)
			{
				// New file has lower hash; delete the previous best and replace it.
				PlatformFile.DeleteFile(*Existing->FullPath);
				Existing->Hash = Hash;
				Existing->FullPath = FString(FullPath);
			}
			else
			{
				// Higher or equal hash; delete this file.
				PlatformFile.DeleteFile(FilenameOrDirectory);
			}

			return true;
		});

	// Rename each surviving file to its base path (the final non-suffixed name).
	for (auto& Pair : BestPerBasePath)
	{
		const FString& BasePath = Pair.Key;
		const FString& BestFile = Pair.Value.FullPath;

		PlatformFile.DeleteFile(*BasePath);
		if (!PlatformFile.MoveFile(*BasePath, *BestFile))
		{
			UE_LOGF(LogShaderSymbolExport, Warning, "Failed to rename deterministic symbol file to \"%ls\".", *BasePath);
		}
	}

	if (BestPerBasePath.Num() > 0)
	{
		UE_LOGF(LogShaderSymbolExport, Display, "Coalesced %d deterministic %ls shader symbol files to lowest-hash variants.", BestPerBasePath.Num(), *ShaderFormatStr);
	}
}

void FShaderSymbolExport::ExportSymbols(TConstArrayView<FShaderSymbolFile> SymbolFiles, const FString& DebugInfo)
{
	// Double-checked locking: atomic acquire load on the fast path, InitCs taken only on first call
	if (!bInitialized.load(std::memory_order_acquire))
	{
		FScopeLock Lock(&InitCs);
		if (!bInitialized.load(std::memory_order_relaxed))
		{
			Initialize();
			bInitialized.store(true, std::memory_order_release);
		}
	}

	if (!bCanExport)
	{
		return;
	}

	// Copy the symbol files for the async task (they use FCompressedBuffer which is ref-counted, so this is cheap).
	TArray<FShaderSymbolFile> SymbolFilesCopy(SymbolFiles);

	PendingTaskCount.fetch_add(1, std::memory_order_relaxed);

	UE::Tasks::Launch(TEXT("ShaderSymbolExport::WriteSymbolData"),
		[this, SymbolFilesCopy = MoveTemp(SymbolFilesCopy), DebugInfo]()
		{
			for (const FShaderSymbolFile& File : SymbolFilesCopy)
			{
				FSharedBuffer Decompressed = File.HasData() ? File.CompressedContents.Decompress() : FSharedBuffer();
				WriteSymbolData(File.Name, DebugInfo, MakeArrayView(reinterpret_cast<const uint8*>(Decompressed.GetData()), static_cast<int32>(Decompressed.GetSize())));
			}

			if (PendingTaskCount.fetch_sub(1, std::memory_order_acq_rel) == 1)
			{
				AllTasksDoneEvent->Trigger();
			}
		},
		UE::Tasks::ETaskPriority::BackgroundLow);
}

void FShaderSymbolExport::Finalize()
{
	// Wait for all in-flight write tasks to complete before finalizing artifacts.
	AllTasksDoneEvent->Reset();
	if (PendingTaskCount.load(std::memory_order_acquire) > 0)
	{
		AllTasksDoneEvent->Wait();
	}

	// In deterministic mode, coalesce all hash-suffixed loose files to the lowest-hash variant.
	// Only the multiprocess owner coalesces (in single-process mode, this is always true).
	if (IsDeterministic() && bMultiprocessOwner)
	{
		CoalesceDeterministicSymbols();
	}

	// Used to match worker files, for example (if not overridden): [ShaderSymbols_]N.ext
	const FString WorkerName = ExportFileName + TEXT("_");

	if (ShaderInfos.Num())
	{
		if (InfoFilePath.Len())
		{
			IFileManager& FileManager = IFileManager::Get();
			if (bMultiprocessOwner)
			{
				// If we are the multiprocess owner merge in any other files we find which have the same FileName, outside of multiproc numbering
				// We do not want to delete other unrelated files in the export path, potentially from other cook jobs
				// We will chunk up the worker files into {Hash, Data} pairs, dedupe them with ours, and sort them all
				IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
				TArray<FString> FilesToMergeIn;
				PlatformFile.FindFiles(FilesToMergeIn, *ExportPath, InfoFileExtension);
				FilesToMergeIn.RemoveAll([&WorkerName](const FString& FileName)
					{
						return !FPathViews::GetPathLeaf(FileName).StartsWith(WorkerName);
					});

				for (const FString& InfoFile : FilesToMergeIn)
				{
					TUniquePtr<FArchive> Reader = TUniquePtr<FArchive>(FileManager.CreateFileReader(*InfoFile));
					if (Reader.IsValid())
					{
						int64 Size = Reader->TotalSize();
						TArray<uint8> RawData;
						RawData.AddUninitialized(Size);
						Reader->Serialize(RawData.GetData(), Size);
						Reader->Close();

						TArray<FString> Lines;
						FString(StringCast<TCHAR>(reinterpret_cast<const ANSICHAR*>(RawData.GetData())).Get()).ParseIntoArrayLines(Lines);

						for (const FString& Line : Lines)
						{
							int32 Space;
							Line.FindChar(TEXT(' '), Space);
							if (Space != INDEX_NONE)
							{
								FString Filename = Line.Left(Space);

								// if this symbol is new to the multiproc owner, store it
								bool bAlreadyInSet = false;
								ExportedShaders.Add(Filename, &bAlreadyInSet);
								if (bAlreadyInSet)
								{
									// The multiproc owner has already seen this hash
									DuplicateSymbols++;
								}
								else
								{
									FString DebugData = Line.Right(Line.Len() - Space - 1);
									ShaderInfos.Add({ Filename, DebugData });
								}
							}
						}
					}
					PlatformFile.DeleteFile(*InfoFile);
				}
			}

			// sort and combine the data for output
			ShaderInfos.KeySort([](const FString& A, const FString& B) { return A < B; });

			TArray<uint8> Output;
			for (TPair<FString, FString> Info : ShaderInfos)
			{
				auto TmpHash = StringCast<ANSICHAR>(*Info.Key);
				auto TmpData = StringCast<ANSICHAR>(*Info.Value);
				Output.Append((const uint8*)TmpHash.Get(), TmpHash.Length());
				Output.Add(' ');
				Output.Append((const uint8*)TmpData.Get(), TmpData.Length());
				Output.Add('\n');
			}

			TUniquePtr<FArchive> Writer = TUniquePtr<FArchive>(FileManager.CreateFileWriter(*InfoFilePath));
			if (Writer.IsValid())
			{
				Writer->Serialize(Output.GetData(), Output.Num());
				Writer->Close();
				UE_LOGF(LogShaderSymbolExport, Display, "Wrote %d records into shader symbols info output file \"%ls\".", ShaderInfos.Num(), *InfoFilePath);
				const uint32 NumDuplicates = DuplicateSymbols.load(std::memory_order_relaxed);
				const uint32 TotalShaders = ShaderInfos.Num() + NumDuplicates;
				UE_LOGF(LogShaderSymbolExport, Display, "%ls %d total shaders, %d shaders after platform deduplication. %d duplicates (%4.1f%%).",
					*ShaderFormatStr,
					TotalShaders, TotalShaders - NumDuplicates, NumDuplicates, (float)NumDuplicates / TotalShaders * 100.0f);
			}
			else
			{
				UE_LOGF(LogShaderSymbolExport, Error, "Failed to create shader symbols output file \"%ls\".", *InfoFilePath);
			}
		}
	}

	if (ZipWriter && bMultiprocessOwner)
	{
		IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
		TArray<FString> ZipsToMergeIn;
		PlatformFile.FindFiles(ZipsToMergeIn, *ExportPath, ZipFileExtension);
		ZipsToMergeIn.RemoveAll([&WorkerName](const FString& FileName)
			{
				return !FPathViews::GetPathLeaf(FileName).StartsWith(WorkerName);
			});

		for (const FString& ZipFile : ZipsToMergeIn)
		{
			{
				FZipArchiveReader Reader(PlatformFile.OpenRead(*ZipFile));
				bool bAllValid = false;
				if (Reader.IsValid())
				{
					bAllValid = true;
					for (const FString& EmbeddedFileName : Reader.GetFileNames())
					{
						TArray<uint8> Contents;
						if (!Reader.TryReadFile(EmbeddedFileName, Contents))
						{
							bAllValid = false;
							continue;
						}
						ZipWriter->AddFile(EmbeddedFileName, Contents, FDateTime::Now());
					}
				}
				if (!bAllValid)
				{
					UE_LOGF(LogShaderSymbolExport, Error,
						"Failed to read from CookWorker shader symbols output file \"%ls\". Some shader symbols will be missing.",
						*ZipFile);
				}
			}
			PlatformFile.DeleteFile(*ZipFile);
		}
	}
	ZipWriter.Reset();
}

#endif // WITH_EDITOR
