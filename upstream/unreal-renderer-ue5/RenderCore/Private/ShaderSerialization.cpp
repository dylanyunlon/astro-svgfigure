// Copyright Epic Games, Inc. All Rights Reserved.

#include "ShaderSerialization.h"

#include "DerivedDataCacheKey.h"
#include "DerivedDataCacheRecord.h"
#include "DerivedDataValue.h"
#include "Serialization/CompactBinaryWriter.h"
#include "Serialization/MemoryReader.h"
#include "Serialization/MemoryWriter.h"
#include "ShaderCore.h"

FShaderCacheSerializeContext::FShaderCacheSerializeContext()
{
	ShaderCode = OwnedShaderCode;
	ShaderSymbolFiles = OwnedShaderSymbolFiles;

	checkf(ShaderCode.Num() == ShaderSymbolFiles.Num(), TEXT("It is required to serialize a (possibly empty, but non-null) symbol file array for every code buffer."));
}

FShaderCacheSerializeContext::~FShaderCacheSerializeContext() = default;

void FShaderCacheSerializeContext::MoveCode(TArray<FCompositeBuffer>& TargetCode, TArray<TArray<FShaderSymbolFile>>& TargetSymbolFiles)
{
	TargetCode = MoveTemp(OwnedShaderCode);
	ShaderCode = TargetCode;
	TargetSymbolFiles = MoveTemp(OwnedShaderSymbolFiles);
	ShaderSymbolFiles = TargetSymbolFiles;

	checkf(ShaderCode.Num() == ShaderSymbolFiles.Num(), TEXT("It is required to serialize a (possibly empty, but non-null) symbol file array for every code buffer."));
}

FShaderCacheSaveContext::FShaderCacheSaveContext()
{
	Reset();
}

FShaderCacheSaveContext::~FShaderCacheSaveContext() = default;
FShaderCacheLoadContext::FShaderCacheLoadContext() = default;
FShaderCacheLoadContext::~FShaderCacheLoadContext() = default;

#if WITH_EDITOR
const UE::DerivedData::FValueId ShaderObjectDataValue = UE::DerivedData::FValueId::FromName(TEXT("ShaderObjectData"));
const UE::DerivedData::FValueId ShaderCodeDataValue = UE::DerivedData::FValueId::FromName(TEXT("ShaderCodeData"));
const UE::DerivedData::FValueId ShaderSymbolsDataValue = UE::DerivedData::FValueId::FromName(TEXT("ShaderSymbolsData"));
const FAnsiStringView CodeCountMetaField = ANSITEXTVIEW("CodeCount");
const FAnsiStringView HasSymbolsMetaField = ANSITEXTVIEW("bHasSymbols");
#endif

void FShaderCacheSaveContext::SerializeCode(FShaderCodeResource& Resource, int32 Index)
{
	OwnedShaderCode.Add(Resource.GetCacheBuffer());
	// reset the array view any time an entry is added; we do this instead of calling Resize in the reserve delegate
	// and setting it there since not all code paths (i.e. single job cache records) call reserve
	ShaderCode = OwnedShaderCode;

	OwnedShaderSymbolFiles.Add(TArray<FShaderSymbolFile>(Resource.GetSymbolFiles()));
	ShaderSymbolFiles = OwnedShaderSymbolFiles;

	checkf(ShaderCode.Num() == ShaderSymbolFiles.Num(), TEXT("It is required to serialize a (possibly empty, but non-null) symbol file array for every code buffer."));
}

void FShaderCacheSaveContext::ReserveCode(int32 Count)
{
	OwnedShaderCode.Reserve(Count);
	OwnedShaderSymbolFiles.Reserve(Count);
}

void FShaderCacheSaveContext::Reset()
{
	ShaderObjectData.Reset();
	OwnedShaderCode.Reset();
	OwnedShaderSymbolFiles.Reset();
	Writer = MakeUnique<FMemoryWriter64>(ShaderObjectRawData, /*bIsPersistent=*/ true);
	Ar = Writer.Get();
}

void FShaderCacheSaveContext::Finalize()
{
	if (!ShaderObjectData)
	{
		ShaderObjectData = MakeSharedBufferFromArray(MoveTemp(ShaderObjectRawData));
	}
}

#if WITH_EDITOR
UE::DerivedData::FCacheRecord FShaderCacheSaveContext::BuildCacheRecord(const UE::DerivedData::FCacheKey& Key)
{
	Finalize();

	UE::DerivedData::FCacheRecordBuilder RecordBuilder(Key);
	RecordBuilder.AddValue(ShaderObjectDataValue, ShaderObjectData);
	int32 CodeIndex = 0;
	// Code buffers and per-file symbol payloads are already compressed, don't waste cycles attempting (and failing) to recompress them
	const ECompressedBufferCompressor SkipCompression = ECompressedBufferCompressor::NotSet;
	const ECompressedBufferCompressionLevel SkipCompressionLevel = ECompressedBufferCompressionLevel::None;

	// Use a meta field to indicate whether we are adding symbol values to this record; we do so to prevent per-value overhead of symbol buffers if they are empty
	// (each value in a cache record has a 64 byte overhead, which can be significant when we're pushing millions of individual shader cache records)
	bool bHasSymbols = false;
	for (const TArray<FShaderSymbolFile>& Files : ShaderSymbolFiles)
	{
		if (Files.Num() > 0)
		{
			bHasSymbols = true;
			break;
		}
	}

	for (FCompositeBuffer& CodeBuf : ShaderCode)
	{
		RecordBuilder.AddValue(ShaderCodeDataValue.MakeIndexed(CodeIndex), UE::DerivedData::FValue(FCompressedBuffer::Compress(CodeBuf, SkipCompression, SkipCompressionLevel)));

		if (bHasSymbols)
		{
			// Serialize each shader's symbol files into a single value: a serialized TArray<FShaderSymbolFile>
			TArray<uint8> SymbolFilesBlob;
			FMemoryWriter BlobWriter(SymbolFilesBlob, /*bIsPersistent*/ true);
			BlobWriter << ShaderSymbolFiles[CodeIndex];
			RecordBuilder.AddValue(ShaderSymbolsDataValue.MakeIndexed(CodeIndex),
				UE::DerivedData::FValue(FCompressedBuffer::Compress(
					FSharedBuffer::Clone(SymbolFilesBlob.GetData(), SymbolFilesBlob.Num()),
					SkipCompression, SkipCompressionLevel)));
		}

		CodeIndex++;
	}

	TCbWriter<16> MetaWriter;
	MetaWriter.BeginObject();
	MetaWriter.AddInteger(CodeCountMetaField, ShaderCode.Num());
	MetaWriter.AddBool(HasSymbolsMetaField, bHasSymbols);
	MetaWriter.EndObject();

	RecordBuilder.SetMeta(MetaWriter.Save().AsObject());

	return RecordBuilder.Build();
}
#endif


FShaderCacheLoadContext::FShaderCacheLoadContext(FSharedBuffer InShaderObjectData, TArrayView<FCompositeBuffer> InCodeBuffers, TArrayView<TArray<FShaderSymbolFile>> InSymbolFiles)
{
	Reset(InShaderObjectData, InCodeBuffers, InSymbolFiles);
}

void FShaderCacheLoadContext::Reset(FSharedBuffer InShaderObjectData, TArrayView<FCompositeBuffer> InCodeBuffers, TArrayView<TArray<FShaderSymbolFile>> InSymbolFiles)
{
	ShaderObjectData = InShaderObjectData;
	ShaderCode = InCodeBuffers;
	ShaderSymbolFiles = InSymbolFiles;

	checkf(ShaderCode.Num() == ShaderSymbolFiles.Num(), TEXT("It is required to serialize a (possibly empty, but non-null) symbol file array for every code buffer."));

	Reader = MakeUnique<FMemoryReaderView>(ShaderObjectData, /*bIsPersistent=*/ true);
	Ar = Reader.Get();
}

void FShaderCacheLoadContext::SerializeCode(FShaderCodeResource& Resource, int32 Index)
{
	// intentional copy of ShaderSymbolFiles; PopulateFromComposite moves this array into the result resource
	// (source array needs to remain intact as it's coming directly from the cached data)
	Resource.PopulateFromComposite(ShaderCode[Index], ShaderSymbolFiles[Index]);
}

void FShaderCacheLoadContext::Reuse()
{
	Reader->Seek(0u);
}

#if WITH_EDITOR
void FShaderCacheLoadContext::ReadFromRecord(const UE::DerivedData::FCacheRecord& Record)
{
	ShaderObjectData = Record.GetValue(ShaderObjectDataValue).GetData().Decompress();

	// Must initialize a memory reader (and the base class archive pointer) after reading the base shadermap data buffer
	// from the DDC record
	Reader = MakeUnique<FMemoryReaderView>(ShaderObjectData, /*bIsPersistent=*/ true);
	Ar = Reader.Get();

	int32 CodeCount = Record.GetMeta()[CodeCountMetaField].AsInt32();
	const bool bHasSymbols = Record.GetMeta()[HasSymbolsMetaField].AsBool();
	OwnedShaderCode.Reserve(CodeCount);
	OwnedShaderSymbolFiles.Reserve(CodeCount);

	for (int32 CodeIndex = 0; CodeIndex < CodeCount; ++CodeIndex)
	{
		FSharedBuffer CombinedBuffer = Record.GetValue(ShaderCodeDataValue.MakeIndexed(CodeIndex)).GetData().Decompress();
		OwnedShaderCode.Add(FShaderCodeResource::Unpack(CombinedBuffer));

		if (bHasSymbols)
		{
			FSharedBuffer SymbolsBlob = Record.GetValue(ShaderSymbolsDataValue.MakeIndexed(CodeIndex)).GetData().Decompress();
			TArray<FShaderSymbolFile> Files;
			FMemoryReaderView BlobReader(SymbolsBlob, /*bIsPersistent*/ true);
			BlobReader << Files;
			OwnedShaderSymbolFiles.Add(MoveTemp(Files));
		}
		else
		{
			OwnedShaderSymbolFiles.AddDefaulted();
		}
	}
	ShaderCode = OwnedShaderCode;
	ShaderSymbolFiles = OwnedShaderSymbolFiles;
}
#endif

