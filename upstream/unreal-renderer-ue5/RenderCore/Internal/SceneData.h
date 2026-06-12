// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "PersistentPrimitiveIndex.h"
#include "Tasks/Task.h"
#include "Templates/MemoryOps.h"
#include "RenderGraphBuilder.h"
#include "SceneRenderingAllocator.h"
#include "RenderUtils.h"
#include "Async/InheritedContext.h"
#include "Misc/GeneratedTypeName.h"

#include <tuple>

#define SCENE_DATA_DEBUG_ACCESS_RACES_DYNAMIC 0

// This define enables full validation, which includes checks that are too costly to run in shipping builds.
// Notably, access outside views (i.e., using the [] operator directly on the data arrays).
#define SCENE_DATA_VALIDATE_ACCESS_CONTEXT (DO_CHECK)

#define SCENE_DATA_FORCEINLINE FORCEINLINE
// Swap for debugging
// #define SCENE_DATA_FORCEINLINE FORCENOINLINE

// General TODO:
// * Make sure things are protected & friends as needed.

namespace SceneData
{

class FDependencyManager;


enum class EProcessingStage
{
	// outside, there is no copy-on-write and all reads and writes trivially go to the backing storage.
	// Dirty tracking is active but since CoW is not, writes will trigger a warning if CoW is enabled for the given array.
	None,
	// In the PreUpdate stage the scene has not yet been modified, but, deleted items have been marked as such.
	PreUpdate,
	// In the update stage, allocations have been updated and copy-on-write is active for all requesting data arrays. It is possible to read the previous values of items if they exist.
	Update,
	// In this stage, read only access is not verified against the root access context, which means that any task can perform read access without triggering a check and without tracking dependencies.
	ReadOnlyUntracked,
	MAX
};

/** 
 */
class FBaseAccessContext
{
public:
	enum class ESyncMode
	{
		ValidateOnly,
		SyncAndValidate,
	};

	RENDERCORE_API explicit FBaseAccessContext(FDependencyManager& InBuilder);

	struct FChildTag {};
	RENDERCORE_API explicit FBaseAccessContext(FDependencyManager& InBuilder, FChildTag);

	FBaseAccessContext(const FBaseAccessContext&) = delete;
	void operator=(const FBaseAccessContext&) = delete;
	FBaseAccessContext(FBaseAccessContext&&) = default;

	SCENE_DATA_FORCEINLINE bool IsRoot() const { return ContextId == 1u; }

	RENDERCORE_API void BeginReadAccess(int32 ResourceId, ESyncMode SyncMode = ESyncMode::SyncAndValidate);
	SCENE_DATA_FORCEINLINE void EndReadAccess(int32 /*ResourceId*/) {}
	
	RENDERCORE_API void BeginWriteAccess(int32 ResourceId, ESyncMode SyncMode = ESyncMode::SyncAndValidate);
	SCENE_DATA_FORCEINLINE void EndWriteAccess(int32 /*ResourceId*/) {}

	uint32           GetContextId()       const { return ContextId; }
	EProcessingStage GetProcessingStage() const { return ProcessingStage; }

	struct FReadScope
	{
		SCENE_DATA_FORCEINLINE explicit FReadScope(FBaseAccessContext& InContext, int32 InResourceId, ESyncMode SyncMode = ESyncMode::SyncAndValidate)
			: Context(InContext)
			, ResourceId(InResourceId)
		{
			Context.BeginReadAccess(ResourceId, SyncMode);
		}
		~FReadScope()
		{
			Context.EndReadAccess(ResourceId);
		}

		FBaseAccessContext& Context;
		int32 ResourceId;
	};

	struct FWriteScope
	{
		SCENE_DATA_FORCEINLINE explicit FWriteScope(FBaseAccessContext& InContext, int32 InResourceId, ESyncMode SyncMode = ESyncMode::SyncAndValidate)
			: Context(InContext)
			, ResourceId(InResourceId)
		{
			Context.BeginWriteAccess(ResourceId, SyncMode);
		}
		~FWriteScope()
		{
			Context.EndWriteAccess(ResourceId);
		}

		FBaseAccessContext& Context;
		int32 ResourceId;
	};


	TBitArray<TInlineAllocator<8>> EnabledReaders;
	TBitArray<TInlineAllocator<8>> EnabledWriters;

	FDependencyManager& Builder;
	uint32 ContextId = 1u;
	EProcessingStage ProcessingStage = EProcessingStage::MAX;

protected:
	friend FDependencyManager;
};

template <typename AllocatorType = FDefaultAllocator>
struct TChunkBitMask
{
	static constexpr int32  BitsPerWord     = 64;
	static constexpr int32  Log2BitsPerWord = 6;
	static constexpr uint64 BitInWordMask   = BitsPerWord - 1;
	static_assert((1 << Log2BitsPerWord) == BitsPerWord, "Log2BitsPerWord must match BitsPerWord");

	static constexpr int32  WordIndex(int32 Bit)  { return Bit >> Log2BitsPerWord; }
	static constexpr int32  BitInWord(int32 Bit)  { return Bit & (int32)BitInWordMask; }
	static constexpr uint64 BitMaskFor(int32 Bit) { return 1ULL << BitInWord(Bit); }

	TArray<uint64, AllocatorType> Words;
	int32 NumBits = 0;

	int32 NumChunks() const { return Words.Num(); }
	int32 Num() const       { return NumBits; }
	bool  IsEmpty() const   { return NumBits == 0; }

	bool IsValidIndex(int32 Bit) const { return Bit >= 0 && Bit < NumBits; }

	bool operator[](int32 Bit) const
	{
		check(IsValidIndex(Bit));
		return (Words[WordIndex(Bit)] & BitMaskFor(Bit)) != 0ULL;
	}

	uint64 GetChunkWord(int32 ChunkIndex) const
	{
		return Words.IsValidIndex(ChunkIndex) ? Words[ChunkIndex] : 0ULL;
	}

	void SetBit(int32 Bit)
	{
		check(IsValidIndex(Bit));
		Words[WordIndex(Bit)] |= BitMaskFor(Bit);
	}

	void ClearBit(int32 Bit)
	{
		check(IsValidIndex(Bit));
		Words[WordIndex(Bit)] &= ~BitMaskFor(Bit);
	}

	void Add(bool bValue, int32 Count = 1)
	{
		PadToNum(NumBits + Count, bValue);
	}

	void PadToNum(int32 InNumBits, bool bValue)
	{
		if (InNumBits <= NumBits)
		{
			return;
		}
		const int32 NewNumWords = FMath::DivideAndRoundUp(InNumBits, BitsPerWord);
		const int32 OldNumWords = Words.Num();
		if (NewNumWords > OldNumWords)
		{
			Words.SetNumZeroed(NewNumWords);
		}
		if (bValue)
		{
			for (int32 Bit = NumBits; Bit < InNumBits; ++Bit)
			{
				Words[WordIndex(Bit)] |= BitMaskFor(Bit);
			}
		}
		NumBits = InNumBits;
	}

	void SetNum(int32 InNumBits, bool bValue)
	{
		const int32 OldNumBits = NumBits;
		if (InNumBits < OldNumBits)
		{
			const int32 NewNumWords = FMath::DivideAndRoundUp(InNumBits, BitsPerWord);
			if (NewNumWords < Words.Num())
			{
				Words.SetNum(NewNumWords);
			}
			// Mask off any tail bits within the now-last word that fall past InNumBits.
			if (NewNumWords > 0)
			{
				const int32 BitsInLastWord = InNumBits - (NewNumWords - 1) * BitsPerWord;
				if (BitsInLastWord < BitsPerWord)
				{
					const uint64 KeepMask = (1ULL << BitsInLastWord) - 1ULL;
					Words[NewNumWords - 1] &= KeepMask;
				}
			}
			NumBits = InNumBits;
		}
		else if (InNumBits > OldNumBits)
		{
			PadToNum(InNumBits, bValue);
		}
	}

	void Reset()
	{
		Words.Reset();
		NumBits = 0;
	}

	int32 FindFrom(bool bValue, int32 Start, int32 End = MAX_int32) const
	{
		const int32 ClampedEnd = FMath::Min(End, NumBits);
		if (Start >= ClampedEnd)
		{
			return INDEX_NONE;
		}
		int32 ChunkIndex = WordIndex(Start);
		const int32 EndChunk = FMath::DivideAndRoundUp(ClampedEnd, BitsPerWord);
		const uint64 Invert = bValue ? 0ULL : ~0ULL;

		// First (partial) word: mask out bits below Start.
		uint64 Word = (Words[ChunkIndex] ^ Invert) & (~0ULL << BitInWord(Start));
		while (true)
		{
			if (Word != 0ULL)
			{
				const int32 Bit = ChunkIndex * BitsPerWord + (int32)FMath::CountTrailingZeros64(Word);
				return Bit < ClampedEnd ? Bit : INDEX_NONE;
			}
			++ChunkIndex;
			if (ChunkIndex >= EndChunk)
			{
				return INDEX_NONE;
			}
			Word = Words[ChunkIndex] ^ Invert;
		}
	}

	int32 FindLastSetBit() const
	{
		for (int32 ChunkIndex = Words.Num() - 1; ChunkIndex >= 0; --ChunkIndex)
		{
			const uint64 Word = Words[ChunkIndex];
			if (Word != 0ULL)
			{
				const int32 BitOffset = BitsPerWord - 1 - (int32)FMath::CountLeadingZeros64(Word);
				return ChunkIndex * BitsPerWord + BitOffset;
			}
		}
		return INDEX_NONE;
	}

	int32 FindAndSetFirstZeroBit(int32 StartBit = 0)
	{
		if (StartBit >= NumBits)
		{
			return INDEX_NONE;
		}
		int32 ChunkIndex = WordIndex(StartBit);
		const int32 EndChunk = FMath::DivideAndRoundUp(NumBits, BitsPerWord);

		uint64 Inverted = ~Words[ChunkIndex] & (~0ULL << BitInWord(StartBit));
		while (true)
		{
			if (Inverted != 0ULL)
			{
				const int32 Bit = ChunkIndex * BitsPerWord + (int32)FMath::CountTrailingZeros64(Inverted);
				if (Bit >= NumBits)
				{
					return INDEX_NONE;
				}
				Words[ChunkIndex] |= BitMaskFor(Bit);
				return Bit;
			}
			++ChunkIndex;
			if (ChunkIndex >= EndChunk)
			{
				return INDEX_NONE;
			}
			Inverted = ~Words[ChunkIndex];
		}
	}

	int32 CountSetBits() const
	{
		int32 Total = 0;
		for (uint64 Word : Words)
		{
			Total += (int32)FMath::CountBits64(Word);
		}
		return Total;
	}

	SIZE_T GetAllocatedSize() const { return Words.GetAllocatedSize(); }

	struct FSetBitIterator
	{
		const TChunkBitMask* Mask;
		int32  ChunkIndex   = 0;
		uint64 CurrentBits  = 0ULL;
		int32  CachedIndex  = INDEX_NONE;

		explicit FSetBitIterator(const TChunkBitMask& InMask)
			: Mask(&InMask)
		{
			Advance();
		}

		explicit operator bool() const { return CachedIndex != INDEX_NONE; }

		int32 GetIndex() const { return CachedIndex; }

		FSetBitIterator& operator++()
		{
			Advance();
			return *this;
		}

	private:
		void Advance()
		{
			while (true)
			{
				if (CurrentBits != 0ULL)
				{
					const int32 BitOffset = (int32)FMath::CountTrailingZeros64(CurrentBits);
					CurrentBits &= CurrentBits - 1ULL;
					const int32 Bit = (ChunkIndex - 1) * BitsPerWord + BitOffset;
					if (Bit < Mask->NumBits)
					{
						CachedIndex = Bit;
						return;
					}
					continue;
				}
				if (ChunkIndex >= Mask->Words.Num())
				{
					CachedIndex = INDEX_NONE;
					return;
				}
				CurrentBits = Mask->Words[ChunkIndex];
				++ChunkIndex;
			}
		}
	};
};

using FChunkBitMask = TChunkBitMask<>;

// Sparse-dirty-mask wrapper used for FBaseDirtyState's AddedMask / DeletedMask / DirtyMasks[].
// Carries a chunk-summary mask (one bit per word of Bits) so TFilteredIdView can precompose
// an IterChunkMask at construction and skip empty chunks without loading the main masks.
struct FTrackedBitArray
{
	FChunkBitMask Bits;
	// Bit C is set iff Bits.Words[C] is non-zero. Composed at filter-view construction time.
	// Since this struct is add-only between Reset() calls, the summary never has false positives.
	FChunkBitMask ChunkSummary;
	int32 Count = 0;

	bool  IsEmpty() const     { return Count == 0; }
	int32 GetCount() const    { return Count; }

	bool IsValidIndex(int32 Bit) const { return Bits.IsValidIndex(Bit); }
	bool operator[](int32 Bit) const   { return Bits[Bit]; }

	void SetBit(int32 Bit, int32 PadTo)
	{
		Bits.PadToNum(PadTo, false);
		ChunkSummary.PadToNum(FMath::DivideAndRoundUp(PadTo, FChunkBitMask::BitsPerWord), false);
		if (!Bits[Bit])
		{
			Bits.SetBit(Bit);
			ChunkSummary.SetBit(FChunkBitMask::WordIndex(Bit));
			++Count;
		}
	}

	void Reset()
	{
		Bits.Reset();
		ChunkSummary.Reset();
		Count = 0;
	}

	SIZE_T GetAllocatedSize() const { return Bits.GetAllocatedSize() + ChunkSummary.GetAllocatedSize(); }
};

class FIdAllocator
{
public:
	// Chunks of 64 is probably good enough
	static constexpr int32 ChunkSize = 64;
	// 4 is good for testing
	//static constexpr int32 ChunkSize = 4;
	static_assert(ChunkSize == FChunkBitMask::BitsPerWord, "FIdAllocator::ChunkSize must equal FChunkBitMask::BitsPerWord so chunk indices map 1:1 to mask words");

	// A TypeId is an integer that must form a semi-compact range, all allocations of a given TypeId are grouped together in chunks of ChunkSize
	// Thus, a consecutive range of IDs modulo ChunkSize are always guaranteed to share type ID.
	int32 Allocate(int32 TypeId)
	{
		if (bShouldConsolidate)
		{
			Consolidate();
		}

		TypeAllocators.PadToNum(TypeId + 1, FTypeAllocator{});
		FTypeAllocator& TypeAlloc = TypeAllocators[TypeId];

		int32 ResultId = INDEX_NONE;
		for(FTypeAllocator::FChunkInfo& ChunkInfo : TypeAlloc.ChunkInfos)
		{
			// There's space in the current chunk
			if (int32(ChunkInfo.Count) < ChunkSize)
			{
				int32 ChunkStartOffset = int32(ChunkInfo.Index) * ChunkSize;
				ResultId = AllocatedIdMask.FindAndSetFirstZeroBit(ChunkStartOffset);
				++ChunkInfo.Count;
				check(ResultId <  ChunkStartOffset + int32(ChunkInfo.Count));
				return ResultId;
			}
		}

		// No chunk found, allocate a new one.
		int32 LocalChunkIndex = TypeAlloc.ChunkInfos.Num();
		FTypeAllocator::FChunkInfo& ChunkInfo = TypeAlloc.ChunkInfos.Emplace_GetRef(AllocateChunkIndex(TypeId, LocalChunkIndex), 1u);
		ResultId = int32(ChunkInfo.Index) * ChunkSize;
		// Mark as allocated
		check(!AllocatedIdMask[ResultId]);
		AllocatedIdMask.SetBit(ResultId);

		//check(TypeAlloc.ChunkInfos.Num() == TypeAlloc.AllocatedIdMask.Num() / ChunkSize);
		return ResultId;
	}

	void Free(int32 Id)
	{
		// 1. figure out type (or do we require that as parameter?)
		int32 ChunkIndex = Id / ChunkSize;
		FAllocateChunkInfo AllocateChunkInfo = AllocatedChunkInfo[ChunkIndex];
		FTypeAllocator& TypeAlloc = TypeAllocators[AllocateChunkInfo.TypeId];

		check(TypeAlloc.ChunkInfos[AllocateChunkInfo.LocalChunkIndex].Index == ChunkIndex);
		check(TypeAlloc.ChunkInfos[AllocateChunkInfo.LocalChunkIndex].Count > 0u);
		TypeAlloc.ChunkInfos[AllocateChunkInfo.LocalChunkIndex].Count -= 1;

		check(AllocatedIdMask[Id]);
		AllocatedIdMask.ClearBit(Id);

		// auto-consolidate next time an allocation is done
		bShouldConsolidate = true;
	}

	int32 GetTypeId(int32 Id)
	{
		int32 ChunkIndex = Id / ChunkSize;
		return AllocatedChunkInfo[ChunkIndex].TypeId;
	}

	void Consolidate()
	{
		for (FTypeAllocator& TypeAllocator : TypeAllocators)
		{
			// Free chunks that have become entirely empty, returning their global slot so
			// GetMaxAllocatedIdChunked() can shrink and AllocateChunkIndex() can reuse them.
			for (int32 LocalIndex = TypeAllocator.ChunkInfos.Num() - 1; LocalIndex >= 0; --LocalIndex)
			{
				if (TypeAllocator.ChunkInfos[LocalIndex].Count == 0)
				{
					FreeChunk(int32(TypeAllocator.ChunkInfos[LocalIndex].Index));
					TypeAllocator.ChunkInfos.RemoveAtSwap(LocalIndex, EAllowShrinking::No);
				}
			}
			// Maintain sorted invariant (RemoveAtSwap may disrupt ordering).
			TypeAllocator.ChunkInfos.Sort();
			// RemoveAtSwap + Sort can change every surviving entry's position in ChunkInfos.
			// Re-write AllocatedChunkInfo[global].LocalChunkIndex so that Free() can still
			// find the right ChunkInfos entry via AllocatedChunkInfo[Id/ChunkSize].
			for (int32 NewLocalIndex = 0; NewLocalIndex < TypeAllocator.ChunkInfos.Num(); ++NewLocalIndex)
			{
				int32 GlobalChunkIndex = int32(TypeAllocator.ChunkInfos[NewLocalIndex].Index);
				AllocatedChunkInfo[GlobalChunkIndex].LocalChunkIndex = NewLocalIndex;
			}
		}

		bShouldConsolidate = false;
	}

	int32 AllocateChunkIndex(int32 TypeId, int32 LocalChunkIndex)
	{
		int32 ChunkIndex = AllocatedChunkMask.FindAndSetFirstZeroBit();
		if (ChunkIndex == INDEX_NONE)
		{
			ChunkIndex = AllocatedChunkMask.Num();
			AllocatedChunkMask.Add(true);
			AllocatedChunkInfo.Emplace(TypeId, LocalChunkIndex);
			AllocatedIdMask.Add(false, ChunkSize);
		}
		else
		{
			check(AllocatedChunkInfo[ChunkIndex].TypeId == INDEX_NONE);
			AllocatedChunkInfo[ChunkIndex] = FAllocateChunkInfo { TypeId, LocalChunkIndex };
		}
		return ChunkIndex;
	}

	void FreeChunk(int32 ChunkIndex)
	{
		AllocatedChunkMask.ClearBit(ChunkIndex);
		AllocatedChunkInfo[ChunkIndex] = FAllocateChunkInfo{};
	}

	int32 GetMaxAllocatedIdChunked() const
	{
		//
		int32 LastUsedChunk = AllocatedChunkMask.FindLastSetBit();
		static_assert(INDEX_NONE == -1, "We assume INDEX_NONE is -1 such that we can use that value to set the sizes directly");
		return (LastUsedChunk + 1) * ChunkSize;
	}

	int32 GetNumAllocatedIds() const
	{
		return AllocatedIdMask.Num();
	}

	FChunkBitMask AllocatedChunkMask;

	struct FAllocateChunkInfo
	{
		int32 TypeId = INDEX_NONE;
		int32 LocalChunkIndex = INDEX_NONE;
	};
	TArray<FAllocateChunkInfo> AllocatedChunkInfo;

	struct FTypeAllocator
	{
		struct FChunkInfo
		{
			bool operator <(const FChunkInfo& B) const { return Index < B.Index; }
			uint32 Index : 24;
			uint32 Count : 8;
		};
		TArray<FChunkInfo> ChunkInfos;
	};

	SIZE_T GetAllocatedSize() const 
	{
		return AllocatedIdMask.GetAllocatedSize() + TypeAllocators.GetAllocatedSize();
	}


	FChunkBitMask AllocatedIdMask;

	bool bShouldConsolidate = false;
	TArray<FTypeAllocator> TypeAllocators;
};

class FBaseDataManager;

class FBaseDataArray
{
public:
	FBaseDataArray(FBaseDataManager& Manager, const TCHAR* InDataTypeName);
	UE_NONCOPYABLE(FBaseDataArray);

	virtual ~FBaseDataArray() {}

	virtual void UpdateAllocations(int32 Num, const FChunkBitMask& DeletedMask, const FChunkBitMask& AddedMask) = 0;

	/** Merges any pending CoW writes back into committed storage. No-op by default. */
	virtual void CommitCoWWrites() {}

	virtual SIZE_T GetAllocatedSize() const = 0;


	int32 GetRegistrationId() const { return RegistrationId; }


	int32 RegistrationId = INDEX_NONE;
	const TCHAR* DataTypeName = TEXT("");

#if SCENE_DATA_DEBUG_ACCESS_RACES_DYNAMIC
private:
	// Layout: bits 48..63 = writer ctx, 32..47 = single-reader ctx, 16..31 = writer depth,
	// 0..15 = reader depth. All four fields are 16-bit.
	static constexpr uint64 WriterShift      = 48;
	static constexpr uint64 ReaderCtxShift   = 32;
	static constexpr uint64 WriterCountShift = 16;
	static constexpr uint64 CtxMask          = 0xFFFFull;
	static constexpr uint64 CountMask        = 0xFFFFull;

	static uint32 UnpackWriter(uint64 S)        { return uint32((S >> WriterShift)      & CtxMask); }
	static uint32 UnpackReaderContext(uint64 S) { return uint32((S >> ReaderCtxShift)   & CtxMask); }
	static uint32 UnpackWriterCount(uint64 S)   { return uint32((S >> WriterCountShift) & CountMask); }
	static uint32 UnpackReaderCount(uint64 S)   { return uint32(S & CountMask); }

	static uint64 Pack(uint32 W, uint32 RC, uint32 WC, uint32 R)
	{
		return (uint64(W  & CtxMask)   << WriterShift)
			 | (uint64(RC & CtxMask)   << ReaderCtxShift)
			 | (uint64(WC & CountMask) << WriterCountShift)
			 | uint64(R & CountMask);
	}

	mutable std::atomic<uint64> AccessState{0};

	// Atomically read-modify-write AccessState. The transform receives the unpacked current
	// state and must return the desired packed state. It may be invoked repeatedly under
	// contention, so it must be pure (no externally visible side effects beyond the assertions
	// it embeds). ReaderId tracks "the shared id of all current readers, or 0 if the readers
	// have mixed ids" — install on the 0 -> 1 transition, keep across N -> N+1 when the new
	// reader matches, clear when mixed. Promotion (BeginWrite while readers exist) is allowed
	// only when all current readers share the writer's context.
	template <typename TransformFunc>
	void TransitionState(TransformFunc&& Transform) const;

public:
	RENDERCORE_API void BeginRead(uint32 DebugId) const;
	RENDERCORE_API void EndRead(uint32 DebugId) const;
	RENDERCORE_API void BeginWrite(uint32 DebugId);
	RENDERCORE_API void EndWrite(uint32 DebugId);
#else // !SCENE_DATA_DEBUG_ACCESS_RACES_DYNAMIC
	SCENE_DATA_FORCEINLINE void BeginRead(uint32 /*DebugId*/) const {}
	SCENE_DATA_FORCEINLINE void EndRead(uint32 /*DebugId*/) const {}
	SCENE_DATA_FORCEINLINE void BeginWrite(uint32 /*DebugId*/) {}
	SCENE_DATA_FORCEINLINE void EndWrite(uint32 /*DebugId*/) {}
#endif // SCENE_DATA_DEBUG_ACCESS_RACES_DYNAMIC
};

#if SCENE_DATA_DEBUG_ACCESS_RACES_DYNAMIC
RENDERCORE_API uint16& GetDebugIdRef();
RENDERCORE_API uint16  GetEffectiveDebugId();
RENDERCORE_API UE::FInheritedContextExtension& GetDebugIdExtension();
#else
FORCEINLINE uint16 GetEffectiveDebugId() { return 0; }
#endif
class FBaseDataManager
{
public:
	class FBaseDirtyState;
	FBaseDataManager() = default;
	FBaseDataManager(const FBaseDataManager&) = delete;
	FBaseDataManager(FBaseDataManager&& Other) = delete;
	void operator=(FBaseDataManager&& Other) = delete;
	virtual ~FBaseDataManager() {}

	int32 AllocateInternal(int32 TypeId, FBaseDirtyState& DirtyState)
	{
		check(!bLocked);

		int32 Result = IdAllocator.Allocate(TypeId);
		DirtyState.MarkAdded(Result);
		return Result;
	}

	void FreeInternal(int32 Index, FBaseDirtyState& DirtyState)
	{
		check(!bLocked);
		DirtyState.MarkDeleted(Index);
		IdAllocator.Free(Index);
	}

	bool AreAllocationsLocked() const { return bLocked; }

protected:
	void LockAllocationsInternal(const FChunkBitMask& DeletedMask, const FChunkBitMask& AddedMask)
	{
		check(CurrentProcessingStage == EProcessingStage::Update);
		IdAllocator.Consolidate();

		int32 NumSlots = IdAllocator.GetMaxAllocatedIdChunked();
		for (FBaseDataArray* DataArray : DataArrays)
		{
			// Some data arrays are not allocated.
			if (DataArray)
			{
				DataArray->UpdateAllocations(NumSlots, DeletedMask, AddedMask);
			}
		}
	}

	/** Hook fired whenever CurrentProcessingStage transitions. TDataManager overrides this to
	 *  keep its root access context's stage in sync with the manager. */
	virtual void OnProcessingStageChanged() {}

public:

	int32 GetMaxId() const
	{
		check(bLocked);
		return IdAllocator.GetMaxAllocatedIdChunked();
	}

	int32 Num() const
	{
		return IdAllocator.GetNumAllocatedIds();
	}

	/** Returns the allocator's per-ID allocation bit mask (one bit per slot, true = allocated).
	 *  Used by TDataArray<Direct> to find live elements without a separate ValidMask. */
	const FChunkBitMask& GetAllocatedIdMask() const { return IdAllocator.AllocatedIdMask; }

	class FBaseDirtyState
	{
	public:
		FBaseDirtyState(const FBaseDataManager* InOwner) : Owner(InOwner) {}

		void MarkDirty(int32 MemberId, int32 ElementIndex)
		{
			DirtyMasks[MemberId].SetBit(ElementIndex, Owner->IdAllocator.GetMaxAllocatedIdChunked());
		}

		bool IsDirty(int32 MemberId, int32 ElementIndex) const
		{
			return DirtyMasks.IsValidIndex(MemberId)
				&& DirtyMasks[MemberId].IsValidIndex(ElementIndex)
				&& DirtyMasks[MemberId][ElementIndex];
		}

		void MarkDeleted(int32 ElementIndex)
		{
			DeletedMask.SetBit(ElementIndex, Owner->IdAllocator.GetMaxAllocatedIdChunked());
		}

		void MarkAdded(int32 ElementIndex)
		{
			AddedMask.SetBit(ElementIndex, Owner->IdAllocator.GetMaxAllocatedIdChunked());
		}

		void Reset()
		{
			DeletedMask.Reset();
			AddedMask.Reset();
			for (FTrackedBitArray& Mask : DirtyMasks)
			{
				Mask.Reset();
			}
		}

		FTrackedBitArray DeletedMask;
		FTrackedBitArray AddedMask;
		TArray<FTrackedBitArray> DirtyMasks;
		const FBaseDataManager* Owner = nullptr;
	};

	int32 RegisterDataArray(FBaseDataArray* Array)
	{
		int32 RegistrationId = DataArrays.Num();
		DataArrays.Emplace(Array);
		return RegistrationId;
	}

	int32 GetMaxRegistrationId() const
	{
		return DataArrays.Num();
	}

	// CommitAllCoWWrites lives on TDataManager — it needs the access-context type and DirtyState,
	// neither of which is visible at the base.

	/** */
	EProcessingStage CurrentProcessingStage = EProcessingStage::None;


	/** Debug toggle: when true, the data array specializations dump a callstack on every write
	 *  taken outside the update window (CurrentProcessingStage == None).  Useful for finding
	 *  unexpected writers; left off in production builds. */
	RENDERCORE_API static bool bLogOutsideUpdateWrites;

	/** Debug toggle: when true, the dependency manager dumps a callstack each time the root
	 *  context blocks waiting on a pending task (e.g. an in-flight writer that the render thread
	 *  is forced to serialize with). Useful for finding RT stalls; left off in production. */
	RENDERCORE_API static bool bLogRootContextWaits;

	SIZE_T GetAllocatedSize() const
	{
		SIZE_T Total = IdAllocator.GetAllocatedSize();
		for (FBaseDataArray* Array : DataArrays)
		{
			Total += Array->GetAllocatedSize();
		}
		return Total;
	}

	int32 GetNumResources() const { return DataArrays.Num(); }

	int32 GetBaseResourceIdOffset() const { return BaseResourceIdOffset; }

	void SetBaseResourceIdOffset(int32 InBaseResourceIdOffset)
	{
		check((BaseResourceIdOffset == INDEX_NONE) != (InBaseResourceIdOffset == INDEX_NONE));
		BaseResourceIdOffset = InBaseResourceIdOffset;
	}

protected:
	FIdAllocator IdAllocator;
	TArray<FBaseDataArray*> DataArrays;
	bool bLocked = true;
	int32 BaseResourceIdOffset = INDEX_NONE;
};

inline FBaseDataArray::FBaseDataArray(FBaseDataManager& Manager, const TCHAR* InDataTypeName)
	: RegistrationId(Manager.RegisterDataArray(this))
	, DataTypeName(InDataTypeName)
{
}

template <typename ManagerType>
struct TBaseDataArray : public FBaseDataArray
{
public:
	using FDataManager = ManagerType;

	TBaseDataArray(FDataManager& InDataManager, const TCHAR* DataTypeName)
	:	FBaseDataArray(InDataManager, DataTypeName)
	,	DataManager(InDataManager) {}


	void MarkDirty(typename FDataManager::FId Id)
	{
		DataManager.MarkDirty(GetRegistrationId(), Id);
	}

	bool IsDirty(FDataManager::FId Id) const
	{
		return IsValidId(Id) && DataManager.IsDirty(GetRegistrationId(), Id);
	}

	bool IsValidId(FDataManager::FId Id) const
	{
		return DataManager.IsValidIndex(Id);
	}

	/**
	 * Enable Copy-on-write for this data array, use in e.g., PreSceneUpdate for data arrays your extension will need previous data for.
	 */
	void EnableCopyOnWrite() 
	{
		bEnableCopyOnWrite = true;
	}

	bool IsCopyOnWriteEnabled() const { return bEnableCopyOnWrite; }

	FDataManager& DataManager;
	bool bEnableCopyOnWrite = false;
};

enum class EMapping
{
	Direct,
	Sparse,
	DirtyFlag, // Implies no actual data storage.
	NUM
};

/**
 * Read/write array view, will mark elements dirty on access so never use for read-only purposes!
 * Delegates all dirty-marking and element lookup to the owning array's own methods.
 */
template <typename ArrayType>
class TDataArrayWriteView : public FBaseAccessContext::FWriteScope
{
public:
	using FData = typename ArrayType::FData;
	using FId   = typename ArrayType::FDataManager::FId;
	using FAccessContext   = typename ArrayType::FDataManager::FAccessContext;

	explicit TDataArrayWriteView(ArrayType& InArray, FAccessContext& InAccessContext) 
		: FBaseAccessContext::FWriteScope(InAccessContext, InArray.GetRegistrationId()) 
		, Array(InArray)
		, ProcessingStage(InAccessContext.GetProcessingStage())
		, DebugIdCached(SceneData::GetEffectiveDebugId())
	{
		Array.BeginWrite(DebugIdCached);
	}

	~TDataArrayWriteView()
	{
		Array.EndWrite(DebugIdCached);
	}

	SCENE_DATA_FORCEINLINE bool IsValidId(FId Id) const { return Array.IsValidId(Id); }

	SCENE_DATA_FORCEINLINE bool Contains(FId Id) const { return Array.Contains(Id); }

	/**
	 * Note: Access through write view marks the element dirty.
	 * For Sparse arrays this also default-constructs the element if absent (get-or-add semantics).
	 */
	SCENE_DATA_FORCEINLINE FData& operator[](FId Id) { return Array.GetWriteReferenceInternal(Id, ProcessingStage); }

	SCENE_DATA_FORCEINLINE const FChunkBitMask& GetValidMask() const { return Array.GetValidMask(); }

	struct FIterator
	{
		TDataArrayWriteView& View;
		int32 CurrentIndex;

		SCENE_DATA_FORCEINLINE FIterator(TDataArrayWriteView& InView, int32 InIndex)
		: View(InView)
		, CurrentIndex(InIndex)
		{}

		SCENE_DATA_FORCEINLINE bool operator!=(const FIterator& Other) const { return CurrentIndex != Other.CurrentIndex; }

		SCENE_DATA_FORCEINLINE FData& operator*() { return View[FId{CurrentIndex}]; }

		SCENE_DATA_FORCEINLINE FIterator& operator++()
		{
			CurrentIndex = View.GetValidMask().FindFrom(true, CurrentIndex + 1);
			return *this;
		}
	};

	SCENE_DATA_FORCEINLINE FIterator begin() { return FIterator(*this, GetValidMask().FindFrom(true, 0)); }
	SCENE_DATA_FORCEINLINE FIterator end()   { return FIterator(*this, INDEX_NONE); }

private:
	ArrayType& Array;
	EProcessingStage ProcessingStage;
	uint16 DebugIdCached;
};


/**
 * Read-only array view. Delegates all element lookup to the owning array's own methods.
 */
template <typename ArrayType>
class TDataArrayReadView : public FBaseAccessContext::FReadScope
{
public:
	using FData = typename ArrayType::FData;
	using FId   = typename ArrayType::FDataManager::FId;
	using FAccessContext   = typename ArrayType::FDataManager::FAccessContext;

	SCENE_DATA_FORCEINLINE explicit TDataArrayReadView(const ArrayType& InArray, FAccessContext& InAccessContext)
		: FBaseAccessContext::FReadScope(InAccessContext, InArray.GetRegistrationId())
		, Array(InArray)
		, ProcessingStage(InAccessContext.GetProcessingStage())
		, DebugIdCached(SceneData::GetEffectiveDebugId())
	{
		Array.BeginRead(DebugIdCached);
	}

	SCENE_DATA_FORCEINLINE ~TDataArrayReadView()
	{
		Array.EndRead(DebugIdCached);
	}

	SCENE_DATA_FORCEINLINE bool IsValidId(FId Id) const { return Array.IsValidId(Id); }

	SCENE_DATA_FORCEINLINE bool Contains(FId Id) const { return Array.Contains(Id); }

	/**
	 */
	SCENE_DATA_FORCEINLINE const FData& operator[](FId Id) const
	{
		return Array.GetReadReferenceInternal(Id, ProcessingStage);
	}

	SCENE_DATA_FORCEINLINE const FData& GetPreviousValue(FId Id) const
	{
		check(Array.IsCopyOnWriteEnabled());
		return Array.GetPrevReadReferenceInternal(Id, ProcessingStage);
	}

	SCENE_DATA_FORCEINLINE const FChunkBitMask& GetValidMask() const { return Array.GetValidMask(); }

	struct FIterator
	{
		const TDataArrayReadView& View;
		int32 CurrentIndex;

		SCENE_DATA_FORCEINLINE FIterator(const TDataArrayReadView& InView, int32 InIndex)
		: View(InView)
		, CurrentIndex(InIndex)
		{}

		SCENE_DATA_FORCEINLINE bool operator!=(const FIterator& Other) const { return CurrentIndex != Other.CurrentIndex; }

		SCENE_DATA_FORCEINLINE const FData& operator*() const { return View[FId{CurrentIndex}]; }

		SCENE_DATA_FORCEINLINE FIterator& operator++()
		{
			CurrentIndex = View.GetValidMask().FindFrom(true, CurrentIndex + 1);
			return *this;
		}
	};

	SCENE_DATA_FORCEINLINE FIterator begin() const { return FIterator(*this, GetValidMask().FindFrom(true, 0)); }
	SCENE_DATA_FORCEINLINE FIterator end()   const { return FIterator(*this, INDEX_NONE); }

private:
	const ArrayType& Array;
	EProcessingStage ProcessingStage;
	uint16 DebugIdCached;
};

/**
 * Base template, can't be instantiated because of pure virtual, specialized below for specific EMapping mappings.
 */
template <typename DataType, EMapping InSceneDataMapping, typename ManagerType>
struct TDataArray : public TBaseDataArray<ManagerType>
{
};

/**
 * Default chunk-allocator policy: routes through FMemory. Used for long-lived (multi-frame)
 * `Live` storage in TDenseStorage/TSparseStorage.
 */
struct FDefaultChunkAllocator
{
	static void* Malloc(SIZE_T Size, uint32 Alignment)      { return FMemory::Malloc(Size, Alignment); }
	static void  Free  (void* Ptr,  SIZE_T /*Size*/)        { FMemory::Free(Ptr); }
};

/**
 * Transient chunk-allocator policy: routes through the scene-rendering linear arena.
 * Memory is ref-counted per 64 KB block (see SceneRenderingAllocator.h) — `Free` decrements the
 * block's allocation count and reclaims the block when it hits zero. Used for `CopyOnWriteData`
 * snapshots, which only live during the Update window and are released in CommitCoWWrites.
 */
struct FSceneRenderingChunkAllocator
{
	static void* Malloc(SIZE_T Size, uint32 Alignment) { return FSceneRenderingAllocator::Malloc(Size, Alignment); }
	static void  Free(void* Ptr, SIZE_T /*Size*/)      { FSceneRenderingAllocator::Free(Ptr); }
};

/**
 * Chunk-pointer array providing raw-memory backing for the Direct and Sparse TDataArray
 * specializations.  ChunkSize is an explicit power-of-two template parameter; index decomposition
 * uses shift and mask for O(1) random access.
 *
 * The ChunkAllocatorType policy decides where chunk memory comes from — FDefaultChunkAllocator
 * (heap-backed) for long-lived storage, FSceneRenderingChunkAllocator for transient CoW snapshots.
 *
 * Lifetime split:
 *   - TChunkedStorage owns chunk *memory*: allocates chunks (EnsureChunkAllocated), frees them
 *     in ResizeChunks and its destructor, and reports its allocation footprint.
 *   - The caller owns element *lifetime* — it must destruct any constructed elements before
 *     the containing chunk is freed (via ResizeChunks shrink or TChunkedStorage destruction).
 *
 * Noncopyable: copying would alias chunk pointers and double-free on destruction. Move-only.
 */
template <typename DataType, int32 InChunkSize, bool bInLazyAlloc, typename ChunkAllocatorType = FDefaultChunkAllocator>
struct TChunkedStorage
{
	// If bLazyAlloc is true the chunks are initially nullptr, otherwise they are always allocated.
	static constexpr bool bLazyAlloc = bInLazyAlloc;
	static constexpr int32  ChunkSize  = InChunkSize;
	static_assert(ChunkSize > 0 && (ChunkSize & (ChunkSize - 1)) == 0, "TChunkedStorage ChunkSize must be a power of two");
	static constexpr uint32 ChunkShift = FGenericPlatformMath::FloorLog2((uint32)ChunkSize);
	static constexpr int32  ChunkMask  = ChunkSize - 1;
	static constexpr SIZE_T ChunkBytes = SIZE_T(ChunkSize) * sizeof(DataType);

	TChunkedStorage() = default;
	TChunkedStorage(const TChunkedStorage&) = delete;
	TChunkedStorage& operator=(const TChunkedStorage&) = delete;
	TChunkedStorage(TChunkedStorage&&) = default;
	TChunkedStorage& operator=(TChunkedStorage&&) = default;

	~TChunkedStorage()
	{
		for (DataType* Chunk : Chunks)
		{
			ChunkAllocatorType::Free(Chunk, ChunkBytes);
		}
	}

	SCENE_DATA_FORCEINLINE DataType& ElementAt(int32 Index)             { return Chunks[Index >> ChunkShift][Index & ChunkMask]; }
	SCENE_DATA_FORCEINLINE const DataType& ElementAt(int32 Index) const { return Chunks[Index >> ChunkShift][Index & ChunkMask]; }

	SCENE_DATA_FORCEINLINE void EnsureChunkAllocated(int32 ChunkIndex)
	{
		Chunks.PadToNum(ChunkIndex + 1, nullptr);
		if (Chunks[ChunkIndex] == nullptr)
		{
			Chunks[ChunkIndex] = static_cast<DataType*>(ChunkAllocatorType::Malloc(ChunkBytes, alignof(DataType)));
		}
	}

	SCENE_DATA_FORCEINLINE void EnsureAllocated(int32 ItemIndex)
	{
		EnsureChunkAllocated(ItemIndex >> ChunkShift);
	}

	/** Shrink-or-grow the chunk-pointer array. Surplus chunks are freed (caller must have
	 *  destructed any constructed elements within them first). New entries are nullptr-initialized;
	 *  caller allocates them lazily via EnsureChunkAllocated. */
	void SetNum(int32 NewNum, EAllowShrinking AllowShrinking = EAllowShrinking::No)
	{
		// TODO: AllowShrinking
		int32 NewNumChunks = FMath::DivideAndRoundUp(NewNum, ChunkSize);
		for (int32 ChunkIndex = NewNumChunks; ChunkIndex < Chunks.Num(); ++ChunkIndex)
		{
			ChunkAllocatorType::Free(Chunks[ChunkIndex], ChunkBytes);
		}
		int32 OldNumChunks = Chunks.Num();
		Chunks.SetNumZeroed(NewNumChunks, AllowShrinking);

		// Allocate the new chunks if not lazy
		if (!bLazyAlloc)
		{
			for (int32 ChunkIndex = OldNumChunks; ChunkIndex < Chunks.Num(); ++ChunkIndex)
			{
				check(Chunks[ChunkIndex] == nullptr);
				EnsureChunkAllocated(ChunkIndex);
			}
		}
	}

	SIZE_T GetAllocatedSize() const
	{
		SIZE_T Total = 0;
		for (const DataType* Chunk : Chunks)
		{
			Total += Chunk != nullptr ? ChunkBytes : 0;
		}

		return Total;
	}

	/** One raw-memory block per chunk; nullptr means "not allocated". Element lifetime is caller-managed. */
	TArray<DataType*> Chunks;
};


/**
 * Dense live-storage delegate: every allocated slot has a constructed element. UpdateAllocations
 * destructs deleted items in DeletedMask and default-constructs added items in AddedMask; chunks
 * are managed eagerly by the underlying TChunkedStorage (Sparse=false).
 */
template <typename DataType>
struct TDenseStorage
{
	static constexpr EMapping     Mapping   = EMapping::Direct;
	/** Storage chunk size: largest power of two fitting within ~128 KB worth of elements. */
	static constexpr int32        ChunkSize = (int32)(1u << FGenericPlatformMath::FloorLog2(131072u / (uint32)sizeof(DataType)));
	static_assert(ChunkSize >= FIdAllocator::ChunkSize && (ChunkSize % FIdAllocator::ChunkSize) == 0,
		"Dense storage chunk size must be a multiple of FIdAllocator::ChunkSize so every CoW chunk lies within one storage chunk");

	SCENE_DATA_FORCEINLINE DataType&       ElementAt(int32 Index)       { return Chunks.ElementAt(Index); }
	SCENE_DATA_FORCEINLINE const DataType& ElementAt(int32 Index) const { return Chunks.ElementAt(Index); }

	SCENE_DATA_FORCEINLINE void EnsureAllocated(int32 /*Index*/) {}                                   // SetNum covers every slot
	SCENE_DATA_FORCEINLINE bool IsConstructedAt(int32 /*Index*/) const { return true; }               // every allocated slot is alive
	SCENE_DATA_FORCEINLINE void MarkConstructedIfNeeded(int32 /*Index*/) {}                           // ditto — already constructed

	SCENE_DATA_FORCEINLINE int32 Num() const { return NumElems; }

	template <typename FBaseDataManagerType>
	SCENE_DATA_FORCEINLINE const FChunkBitMask& GetValidMask(const FBaseDataManagerType& Mgr) const { return Mgr.GetAllocatedIdMask(); }

	void UpdateAllocations(int32 NewNum, const FChunkBitMask& DeletedMask, const FChunkBitMask& AddedMask)
	{
		// TODO: Make sure the whole loop is optimized out for trivially destructible things - maybe add templated helper or guard around the whole thing.
		for (FChunkBitMask::FSetBitIterator It(DeletedMask); It; ++It)
		{
			DestructItem(&Chunks.ElementAt(It.GetIndex()));
		}

		Chunks.SetNum(NewNum);
		NumElems = NewNum;

		// Default-construct newly-added elements.
		// TODO:
		//  1. Make sure the whole loop is optimized out for trivially constructible things.
		//  2. efficiency problem: double constructing items that are added - how do we fix this?
		for (FChunkBitMask::FSetBitIterator It(AddedMask); It; ++It)
		{
			DefaultConstructItems<DataType>(&Chunks.ElementAt(It.GetIndex()), 1);
		}
	}

	SIZE_T GetAllocatedSize() const { return Chunks.GetAllocatedSize(); }

	TChunkedStorage<DataType, ChunkSize, /*bLazyAlloc=*/false> Chunks;
	int32                                                     NumElems = 0;
};


/**
 * Sparse live-storage delegate: only explicitly-set slots have constructed elements. Per-slot
 * ValidMask tracks constructedness; chunks are allocated lazily on first write and reclaimed in
 * UpdateAllocations when they become empty.
 */
template <typename DataType>
struct TSparseStorage
{
	static constexpr EMapping Mapping   = EMapping::Sparse;
	static constexpr int32    ChunkSize = FIdAllocator::ChunkSize;

	SCENE_DATA_FORCEINLINE DataType&       ElementAt(int32 Index)       { return Chunks.ElementAt(Index); }
	SCENE_DATA_FORCEINLINE const DataType& ElementAt(int32 Index) const { return Chunks.ElementAt(Index); }

	SCENE_DATA_FORCEINLINE void EnsureAllocated(int32 Index) { Chunks.EnsureAllocated(Index); }
	SCENE_DATA_FORCEINLINE bool IsConstructedAt(int32 Index) const { return ValidMask.IsValidIndex(Index) && ValidMask[Index]; }

	SCENE_DATA_FORCEINLINE void MarkConstructedIfNeeded(int32 Index)
	{
		if (!IsConstructedAt(Index))
		{
			DefaultConstructItems<DataType>(&Chunks.ElementAt(Index), 1);
			ValidMask.PadToNum(Index + 1, false);
			ValidMask.SetBit(Index);
		}
	}

	SCENE_DATA_FORCEINLINE int32 Num() const
	{
		// TODO: make efficient
		return ValidMask.CountSetBits();
	}

	template <typename FBaseDataManagerType>
	SCENE_DATA_FORCEINLINE const FChunkBitMask& GetValidMask(const FBaseDataManagerType& /*Mgr*/) const { return ValidMask; }

	void UpdateAllocations(int32 NewNum, const FChunkBitMask& DeletedMask, const FChunkBitMask& AddedMask)
	{
		check(DeletedMask.IsEmpty() || DeletedMask.Num() == ValidMask.Num());

		// Destruct freed elements in Storage.
		for (FChunkBitMask::FSetBitIterator It(DeletedMask); It; ++It)
		{
			const int32 Index = It.GetIndex();
			if (ValidMask.IsValidIndex(Index) && ValidMask[Index])
			{
				DestructItem(&Chunks.ElementAt(Index));
				ValidMask.ClearBit(Index);
			}
		}

		// Verify no live elements remain in surplus chunks, then resize Storage chunk-pointer array.
		{
			const int32 LastSetBit = ValidMask.FindLastSetBit();
			check(LastSetBit == INDEX_NONE || LastSetBit < NewNum);
		}

		Chunks.SetNum(NewNum);
		ValidMask.SetNum(NewNum, false);

		// Free any chunks within the retained range that are now entirely empty.
		// TODO: maybe make more efficient by tracking chunks that are empty for all and pass in.
		for (int32 ChunkIndex = 0; ChunkIndex < Chunks.Chunks.Num(); ++ChunkIndex)
		{
			if (Chunks.Chunks[ChunkIndex] != nullptr)
			{
				const bool bChunkEmpty = ValidMask.GetChunkWord(ChunkIndex) == 0ULL;
				if (bChunkEmpty)
				{
					FMemory::Free(Chunks.Chunks[ChunkIndex]);
					Chunks.Chunks[ChunkIndex] = nullptr;
				}
			}
		}
	}

	SIZE_T GetAllocatedSize() const { return Chunks.GetAllocatedSize() + ValidMask.GetAllocatedSize(); }

	TChunkedStorage<DataType, ChunkSize, /*bLazyAlloc=*/true> Chunks;
	/** One bit per ID slot; true when the element at that slot is constructed in Storage. */
	FChunkBitMask                                            ValidMask;
};


/**
 * Unified TDataArray body shared by both Direct and Sparse mappings. The StorageType delegate
 * (TDenseStorage or TSparseStorage) handles all dense-vs-sparse mechanics (chunk allocation,
 * per-element validity tracking, eager-vs-lazy construction, chunk freeing on shrink). This
 * class owns the CoW shadow plus the view / iteration scaffolding common to both.
 */
template <typename DataType, typename StorageType, typename ManagerType>
struct TDataArrayImpl : public TBaseDataArray<ManagerType>
{
	using FDataManager = ManagerType;
	using FData        = DataType;
	static constexpr EMapping SceneDataMapping = StorageType::Mapping;
	/** CoW dirty granularity (matches FIdAllocator chunk size). */
	static constexpr int32 ChunkSize = FIdAllocator::ChunkSize;
	using FId = typename FDataManager::FId;
	using FAccessContext = typename FDataManager::FAccessContext;

	explicit TDataArrayImpl(ManagerType& InDataManager)
		: TBaseDataArray<ManagerType>(InDataManager, GetGeneratedTypeName<DataType>())
	{}

	using TBaseDataArray<ManagerType>::DataManager;
	using TBaseDataArray<ManagerType>::GetRegistrationId;
	using TBaseDataArray<ManagerType>::MarkDirty;
	using TBaseDataArray<ManagerType>::IsValidId;
	using TBaseDataArray<ManagerType>::bEnableCopyOnWrite;

	~TDataArrayImpl()
	{
		DestructAllCoWSnapshots();
		for (FChunkBitMask::FSetBitIterator It(GetValidMask()); It; ++It)
		{
			DestructItem(&Live.ElementAt(It.GetIndex()));
		}
	}

	/** Returns true iff the slot for this Id holds a constructed live element. For Dense this is
	 *  always true for any allocated Id; for Sparse it reflects the per-slot validity mask. */
	SCENE_DATA_FORCEINLINE bool Contains(FId Id) const
	{
		return Live.IsConstructedAt(DataManager.IdToIndex(Id));
	}

	/**
	 * Read-only access, not synchronized but validated (when checks are enabled).
	 *  - On RenderThread the user must therefore explicitly sync before using the data array.
	 *  - For access in other threads, validation against the access context is performed.
	 *  - Prefer to use GetReadView() and bulk processing.
	 */
	SCENE_DATA_FORCEINLINE const DataType& operator[](FId Id) const
	{
#if SCENE_DATA_VALIDATE_ACCESS_CONTEXT
		FBaseAccessContext::FReadScope Scope(DataManager.ResolveAccessContext(), GetRegistrationId(), FBaseAccessContext::ESyncMode::ValidateOnly);
#endif
		return GetReadReferenceInternal(Id, EProcessingStage::None);
	}

	/**
	 * Get a read view of the data array. This is the preferred method for accessing data.
	 * Validated and (on RenderThread) synchronized access.
	 */
	SCENE_DATA_FORCEINLINE TDataArrayReadView<TDataArrayImpl> GetReadView(FAccessContext& AccessContext) const
	{
		return TDataArrayReadView<TDataArrayImpl>(*this, AccessContext);
	}
	SCENE_DATA_FORCEINLINE TDataArrayReadView<TDataArrayImpl> GetReadView() const { return GetReadView(DataManager.ResolveAccessContext()); }

	/** 
	 * Get write view (marks elements dirty on access). This is the preferred method for accessing data.
	 * Validated and (on RenderThread) synchronized access.
	 */
	SCENE_DATA_FORCEINLINE TDataArrayWriteView<TDataArrayImpl> GetWriteView(FAccessContext& AccessContext)
	{
		return TDataArrayWriteView<TDataArrayImpl>(*this, AccessContext);
	}
	SCENE_DATA_FORCEINLINE TDataArrayWriteView<TDataArrayImpl> GetWriteView() { return GetWriteView(DataManager.ResolveAccessContext()); }

	/**
	 * Get writable reference & mark dirty, not synchronized but validated (when checks are enabled).
	 *  - On RenderThread the user must therefore explicitly sync before using the data array.
	 *  - For access in other threads, validation against the access context is performed.
	 * Prefer bulk processing using GetWriteView().
	 */
	SCENE_DATA_FORCEINLINE DataType& GetWriteReference(FId Id)
	{
#if SCENE_DATA_VALIDATE_ACCESS_CONTEXT
		FBaseAccessContext::FWriteScope Scope(DataManager.ResolveAccessContext(), GetRegistrationId(), FBaseAccessContext::ESyncMode::ValidateOnly);
#endif
		// TODO: EProcessingStage::None means no CoW through this path - possibly not what we want, audit the processing stage thing feels like a semi-useless thing now.
		return GetWriteReferenceInternal(Id, EProcessingStage::None);
	}

	/**
	 * Shorthand for GetWriteReference(Id) = Value.
	 * Prefer bulk processing using GetWriteView().
	 */
	SCENE_DATA_FORCEINLINE void Set(FId Id, const DataType& Value)
	{
		GetWriteReference(Id) = Value;
	}

	SCENE_DATA_FORCEINLINE int32 Num() const { return Live.Num(); }

	/** 
	 */
	SCENE_DATA_FORCEINLINE const FChunkBitMask& GetValidMask() const { return Live.GetValidMask(DataManager); }

	struct FIterator
	{
		const TDataArrayImpl& Array;
		int32 CurrentIndex;

		SCENE_DATA_FORCEINLINE FIterator(const TDataArrayImpl& InArray, int32 InIndex)
		: Array(InArray)
		, CurrentIndex(InIndex)
		{}

		SCENE_DATA_FORCEINLINE bool operator!=(const FIterator& Other) const { return CurrentIndex != Other.CurrentIndex; }
		SCENE_DATA_FORCEINLINE const DataType& operator*() const { return Array[FId{CurrentIndex}]; }
		SCENE_DATA_FORCEINLINE FIterator& operator++()
		{
			CurrentIndex = Array.GetValidMask().FindFrom(true, CurrentIndex + 1);
			return *this;
		}
	};

	SCENE_DATA_FORCEINLINE FIterator begin() const { return FIterator(*this, GetValidMask().FindFrom(true, 0)); }
	SCENE_DATA_FORCEINLINE FIterator end()   const { return FIterator(*this, INDEX_NONE); }

protected:
	template <typename> friend class TDataArrayReadView;
	template <typename> friend class TDataArrayWriteView;

	/**
	 */
	SCENE_DATA_FORCEINLINE const DataType& GetReadReferenceInternal(FId Id, EProcessingStage ProcessingStage) const
	{
		// TODO: ProcessingStage?
		const int32 Index = DataManager.IdToIndex(Id);
		check(Live.IsConstructedAt(Index));
		return Live.ElementAt(Index);
	}

	/**
	 */
	SCENE_DATA_FORCEINLINE const DataType& GetPrevReadReferenceInternal(FId Id, EProcessingStage ProcessingStage) const
	{
		// TODO: ProcessingStage?
		const int32 Index = DataManager.IdToIndex(Id);

		if (UNLIKELY(SceneData::FBaseDataManager::bLogOutsideUpdateWrites) && ProcessingStage != EProcessingStage::Update)
		{
			FDebug::DumpStackTraceToLog(TEXT("Prev value access outside update window"), ELogVerbosity::Warning);
		}

		if (CopyOnWriteMask.IsValidIndex(Index) && CopyOnWriteMask[Index])
		{
			return CopyOnWriteData.ElementAt(Index);
		}
		return Live.ElementAt(Index);
	}

	/** */
	SCENE_DATA_FORCEINLINE DataType& GetWriteReferenceInternal(FId Id, EProcessingStage ProcessingStage)
	{
		const int32 Index = DataManager.IdToIndex(Id);
		Live.EnsureAllocated(Index);                // no-op for dense; allocates chunk for sparse
		DataType& Elem = Live.ElementAt(Index);

		// An item has a previous value iff it's already constructed AND was not just-added in this cycle.
		// Dense: IsConstructedAt is always true once the slot is allocated, so the predicate reduces to !bWasAdded.
		// Sparse: IsConstructedAt reflects ValidMask, so a not-yet-Set just-added slot correctly skips CoW.
		const bool bWasAdded =
			DataManager.DirtyState.AddedMask.IsValidIndex(Index) &&
			DataManager.DirtyState.AddedMask[Index];
		const bool bHasPreviousValue = Live.IsConstructedAt(Index) && !bWasAdded;

		if (ProcessingStage != EProcessingStage::None && bEnableCopyOnWrite && bHasPreviousValue)
		{
			// Prevent writes that trigger CoW during the locked phase
			check(DataManager.AreAllocationsLocked());
			CopyOnWrite(Index, MoveTemp(Elem));
		}

		if (UNLIKELY(SceneData::FBaseDataManager::bLogOutsideUpdateWrites) && ProcessingStage == EProcessingStage::None)
		{
			FDebug::DumpStackTraceToLog(TEXT("TDataArray: write outside update window"), ELogVerbosity::Warning);
		}

		Live.MarkConstructedIfNeeded(Index);        // no-op for dense; lazy default-construct + valid-flag for sparse
		MarkDirty(Id);
		return Elem;
	}


	/** */
	SCENE_DATA_FORCEINLINE void CopyOnWrite(int32 Index, DataType&& Elem)
	{
		check(bEnableCopyOnWrite);

		// Already done
		if (CopyOnWriteMask.IsValidIndex(Index) && CopyOnWriteMask[Index])
		{
			return;
		}

		CopyOnWriteData.EnsureAllocated(Index);
		MoveConstructItems<DataType>(&CopyOnWriteData.ElementAt(Index), &Elem, 1);
		CopyOnWriteMask.PadToNum(Num(), false);
		CopyOnWriteMask[Index] = true;
	}

	/** */
	void DestructAllCoWSnapshots()
	{
		// TODO: Make sure the whole loop is optimized out for trivially destructible things - maybe add templated helper or guard around the whole thing.
		for (TConstSetBitIterator<SceneRenderingBitArrayAllocator> It(CopyOnWriteMask); It; ++It)
		{
			DestructItem(&CopyOnWriteData.ElementAt(It.GetIndex()));
		}
	}

	virtual void UpdateAllocations(int32 NewNum, const FChunkBitMask& DeletedMask, const FChunkBitMask& AddedMask) override
	{
		// CoW state must be empty at update entry: CommitCoWWrites retires snapshots at the end of
		// the previous update window, and writes outside that window (stage == None) skip CoW.
		check(CopyOnWriteMask.IsEmpty());

		Live.UpdateAllocations(NewNum, DeletedMask, AddedMask);
	}

	virtual void CommitCoWWrites() override
	{
		// Live storage already contains the new values; destruct the old-data snapshot and reset.
		DestructAllCoWSnapshots();
		CopyOnWriteData.SetNum(0);
		CopyOnWriteMask.Reset();
	}

	virtual SIZE_T GetAllocatedSize() const override
	{
		return Live.GetAllocatedSize() + CopyOnWriteData.GetAllocatedSize() + CopyOnWriteMask.GetAllocatedSize();
	}

	StorageType Live;
	TChunkedStorage<DataType, ChunkSize, /*bLazyAlloc=*/true, FSceneRenderingChunkAllocator> CopyOnWriteData;
	TBitArray<SceneRenderingBitArrayAllocator>  CopyOnWriteMask;
};


/**
 * Specialization for direct mapping. Thin wrapper over the unified TDataArrayImpl with the dense
 * storage delegate; behavior lives on the impl.
 */
template <typename DataType, typename ManagerType>
struct TDataArray<DataType, EMapping::Direct, ManagerType>
    : public TDataArrayImpl<DataType, TDenseStorage<DataType>, ManagerType>
{
	using Base = TDataArrayImpl<DataType, TDenseStorage<DataType>, ManagerType>;
	using Base::Base;
};

/**
 * Specialization for DirtyFlag mapping & no actual data, only used to create side-effects, doesn't store any data whatever.
 */
template <typename ManagerType>
struct TDataArray<void, EMapping::DirtyFlag, ManagerType> : public TBaseDataArray<ManagerType>
{
	static constexpr EMapping SceneDataMapping = EMapping::DirtyFlag;
	using TBaseDataArray<ManagerType>::MarkDirty;

	explicit TDataArray(ManagerType& InDataManager)
		: TBaseDataArray<ManagerType>(InDataManager, TEXT("DirtyFlag"))
	{}

protected:
	virtual void UpdateAllocations(int32 /*Num*/, const FChunkBitMask& /*DeletedMask*/, const FChunkBitMask& /*AddedMask*/) override { }

	virtual SIZE_T GetAllocatedSize() const override
	{
		return (SIZE_T)0;
	}
};

/**
 * Specialization for sparse mapping. Thin wrapper over the unified TDataArrayImpl with the sparse
 * storage delegate; behavior lives on the impl.
 */
template <typename DataType, typename ManagerType>
struct TDataArray<DataType, EMapping::Sparse, ManagerType>
    : public TDataArrayImpl<DataType, TSparseStorage<DataType>, ManagerType>
{
	using Base = TDataArrayImpl<DataType, TSparseStorage<DataType>, ManagerType>;
	using Base::Base;
};

// TODO: Specialize for bool stored data to use bit array if needed.

template <typename ManagerType> struct TParameters;
template <typename ManagerType> class TDependencyManager;


enum class EAccessFlags : uint32
{
	// ExternalRead = 1u << 0,
	Dependency			= 1u << 1,
	Default				= Dependency,
	None				= 0u,
	RecordAsReadEvent	= 1u << 2,
	// TODO: can implement later, trigger wait when the resource is accessed (requires thread safety, for sub-tasks that share context) 
	// WaitOnRead	 = 0u, // Absence of "Dependency"
};
ENUM_CLASS_FLAGS(EAccessFlags);


struct FOptionalTaskArgs
{
	TArray<UE::Tasks::FTask> Prerequisites = TArray<UE::Tasks::FTask>();
	UE::Tasks::ETaskPriority Priority = UE::Tasks::ETaskPriority::Default;
	bool bCondition = true;
	UE::Tasks::FPipe* Pipe = nullptr;
};

/** 
 */
class FExternalAccessContextContainer
{
	static constexpr int32 MaxManagerTypes = 8;

public:
	template <typename ManagerContextType>
	void Set(ManagerContextType* ManagerContext)
	{
		uint32 TypeIndex = GetManagerTypeIndex<ManagerContextType>();
		check(Entries[TypeIndex] == nullptr);
		Entries[TypeIndex] = ManagerContext;
	}

	template <typename ManagerContextType>
	ManagerContextType* Get() const
	{
		return static_cast<ManagerContextType*>(Entries[GetManagerTypeIndex<ManagerContextType>()]);
	}

private:
	/** Stable per-manager-type slot index into FExternalAccessContextContainer::Entries. */
	template <typename ManagerContextType>
	static uint32 GetManagerTypeIndex()
	{
		static uint32 CachedIndex = TNumericLimits<uint32>::Max();
		if (CachedIndex == TNumericLimits<uint32>::Max())
		{
			CachedIndex = AllocateManagerTypeIndex(GetGeneratedTypeName<ManagerContextType>());
		}
		return CachedIndex;
	}

	RENDERCORE_API static uint32 AllocateManagerTypeIndex(const TCHAR* ManagerTypeName);

	TStaticArray<FBaseAccessContext*, MaxManagerTypes> Entries = {};
};

RENDERCORE_API FExternalAccessContextContainer*& GetExternalAccessContextsRef();
RENDERCORE_API UE::FInheritedContextExtension& GetExternalAccessExtension();

/** 
 * Activates the external-access extension and publishes a pre-built FExternalAccessContextContainer table
 * on the current thread. UE::Tasks' inherited-context capture/restore propagates the table to any
 * task launched while this scope is active. 
 */
struct FExternalAccessTableScope
{
	UE_NONCOPYABLE(FExternalAccessTableScope);

	SCENE_DATA_FORCEINLINE FExternalAccessTableScope(FExternalAccessContextContainer& Contexts)
		: Scope(GetExternalAccessExtension())
		, PrevContexts(GetExternalAccessContextsRef())
	{
		GetExternalAccessContextsRef() = &Contexts;
	}

	SCENE_DATA_FORCEINLINE ~FExternalAccessTableScope()
	{
		GetExternalAccessContextsRef() = PrevContexts;
	}

private:
	UE::FInheritedContextExtensionScope Scope;
	FExternalAccessContextContainer* PrevContexts;
};

/**
 * Pushes a per-thread debug id used by the data-array race detector. The InheritedContext
 * extension propagates the value to any task / ParallelFor launched while the scope is alive.
 *
 *  - Push manually around ParallelFor sites: workers share the launching thread's effective id.
 *  - Used internally by the task wrapper to install the per-task ContextId for the body.
 */
struct FDebugIdScope
{
	UE_NONCOPYABLE(FDebugIdScope);

#if SCENE_DATA_DEBUG_ACCESS_RACES_DYNAMIC
	SCENE_DATA_FORCEINLINE explicit FDebugIdScope(uint16 NewId) : PrevId(GetDebugIdRef())
	{
		GetDebugIdRef() = NewId;
	}

	SCENE_DATA_FORCEINLINE FDebugIdScope() : FDebugIdScope(GetEffectiveDebugId()) {}

	SCENE_DATA_FORCEINLINE ~FDebugIdScope() { GetDebugIdRef() = PrevId; }
private:
	uint16 PrevId;
#else
	SCENE_DATA_FORCEINLINE explicit FDebugIdScope(uint16 /*NewId*/) { }
#endif
};


class FDependencyManager
{
public:
	enum class EWaitState
	{
		None,
		Read,
		Write,
	};

	struct FParameter
	{
		int32 ResourceId = INDEX_NONE;
		EAccessFlags AccessFlags = EAccessFlags::Default;
	};


	struct FResourceState
	{
		int32 LastReadEventId      = INDEX_NONE;
		int32 LastWriteEventId     = INDEX_NONE;
		EWaitState WaitState       = EWaitState::None;
	};

	struct FResourceEvent
	{
		int32 TaskId          = INDEX_NONE;
		int32 PreviousEventId = INDEX_NONE;
		bool  bReadEvent      = true;
	};

	FDependencyManager() = default;
	~FDependencyManager() { Flush(); }

	FDependencyManager(const FDependencyManager&) = delete;
	void operator=(const FDependencyManager&) = delete;

	/**
	 */
	RENDERCORE_API void Flush();

	/** 
	 */
	RENDERCORE_API void FrameEnd();

	RENDERCORE_API bool WaitForResoureRead(int32 ResourceId);
	RENDERCORE_API bool WaitForResoureWrite(int32 ResourceId);

	RENDERCORE_API void ProcessDependencies(
		FBaseAccessContext& ChildDepCtx,
		TArray<UE::Tasks::FTask>& Prerequisites,
		int32 TaskId,
		const TArray<FParameter>& ReadDeps,
		const TArray<FParameter>& WriteDeps,
		bool bIsRunningAsync);

	int32 GetNextTaskId() const { return Tasks.Num(); }
	void AddTask(const UE::Tasks::FTask& Task) { Tasks.Emplace(Task); }

	/** Grows ResourceHeadState so a manager's full registration-id range is covered. Used by
	 *  AddProcessingTask before processing per-resource dependencies. */
	void EnsureResourceHeadStateSize(int32 Num) { ResourceHeadState.PadToNumDefaulted(Num); }

	/** */
	RENDERCORE_API void OnTaskLaunched(const UE::Tasks::FTask& Task);

	/** */
	FSceneRenderingBulkObjectAllocator ChildContextStorage;

private:
	friend class FBaseAccessContext;

	void WaitAndClear(const FResourceEvent& Event);
	void ResetTrackingState();

	TArray<FResourceEvent>   ResourceEvents;
	TArray<FResourceState>   ResourceHeadState;
	TArray<UE::Tasks::FTask> Tasks;
	uint32                   CurrentContextId = 1u;
	int32                    ActiveTaskCount = 0;
	/** 
	 * Suppress checks caused by accessing the root context via non-RT, this allows arbitrary tasks to unsafely access the data for legacy purposes. 
	 * TODO: Remove this when things are converted.
	 */
	static bool bSuppressRootContextValidation;

	// TODO: validate outstanding contexts, std::atomic<int32>       ActiveContextCount = 0;
};

/** Templated derived purely as a type tag — each `TDependencyManager<M>`
 *  exports a distinct nested `FAccessContext` so the compiler rejects cross-manager passes.
 *  It also owns the typed `RootContext` member. All real state lives on the base. */
template <typename ManagerType>
class TDependencyManager : public FDependencyManager
{
public:
	using FOwnerManager = ManagerType;

	class FAccessContext : public FBaseAccessContext
	{
	public:
		using FBaseAccessContext::FBaseAccessContext;
	};

	mutable FAccessContext RootContext{*this};
};


template <typename InDataTraitsType>
class TDataManager : public FBaseDataManager
{
public:
	using FDataTraits = InDataTraitsType;
	using FId = FDataTraits::FIdType;

	// useful alias for derived class - a derived impl
	using FDataManager = TDataManager<FDataTraits>;

	// Helper type aliases
	template <typename DataType>
	using TDataArrayDirect = SceneData::TDataArray<DataType, SceneData::EMapping::Direct, FDataManager>;
	template <typename DataType, SceneData::EMapping Mapping>
	using TDataArray = SceneData::TDataArray<DataType, Mapping, FDataManager>;
	// Dirty flag array, specialized storage type that doesn't actually have any corresponding data
	using FDataArrayDirty = SceneData::TDataArray<void, SceneData::EMapping::DirtyFlag, FDataManager>;
	// Sparse array: only a subset of IDs carry data; elements are constructed lazily on first write.
	template <typename DataType>
	using TDataArraySparse = SceneData::TDataArray<DataType, SceneData::EMapping::Sparse, FDataManager>;
	template <typename DataType>
	using TReadViewSparse = TDataArrayReadView<TDataArrayImpl<DataType, TSparseStorage<DataType>, TDataManager>>;
	template <typename DataType>
	using TReadViewDirect = TDataArrayReadView<TDataArrayImpl<DataType, TDenseStorage<DataType>, TDataManager>>;

	using FDepManager    = TDependencyManager<TDataManager>;
	using FAccessContext = typename FDepManager::FAccessContext;
	using FParameters = TParameters<TDataManager>;

	FDepManager DepManager;

	int32 IdToIndex(FId Id) const
	{
		check(IsValidIndex(Id));
		return FDataTraits::IdToIndex(Id);
	}

	TDataManager()
		: DirtyState(this)
	{
	}

	~TDataManager() = default;

	/**
	 */
	void OnProcessingStageChanged()
	{
		DepManager.RootContext.ProcessingStage = CurrentProcessingStage;
	}

	/**
	 */
	void BeginPreUpdate()
	{
		check(CurrentProcessingStage == EProcessingStage::None || CurrentProcessingStage == EProcessingStage::ReadOnlyUntracked);
		CurrentProcessingStage = EProcessingStage::PreUpdate;
		OnProcessingStageChanged();
	}

	/**
	 */
	void EndPreUpdate()
	{
		check(CurrentProcessingStage == EProcessingStage::PreUpdate);
		CurrentProcessingStage = EProcessingStage::Update;
		OnProcessingStageChanged();
	}

	/**
	 */
	void EndUpdate()
	{
		check(CurrentProcessingStage == EProcessingStage::Update);
		
		// Add a write dep for all arrays, this protectes both the CoW data and the dirty state
		// For the CoW data - we don't need a full write barrier we just need a barrier vs the preceeding tasks that are possibly using it, but we don't need subsequent tasks to see this as a write (because they are not allowed to use CoW data, it is disabled in EProcessingStage::None)
		// For the dirty state - it is the same no subsequently launched work may _read_ the dirty state, and thus would need to add a write barrier to access it.
		auto Params = MakeParameters().Write(*this, EAccessFlags::RecordAsReadEvent | EAccessFlags::Dependency);
		AddProcessingTask(Params, [this]()
		{
			FAccessContext& AccessContext = ResolveAccessContext();
			for (FBaseDataArray* Array : DataArrays)
			{
				FBaseAccessContext::FWriteScope Scope(AccessContext, Array->GetRegistrationId());
				Array->CommitCoWWrites();
			}
			DirtyState.Reset();
		});
		CurrentProcessingStage = EProcessingStage::None;
		OnProcessingStageChanged();
	}

	/** 
	 * Transition to read only stage, after which all read only access is untracked and write access is not permitted.
	 */
	void BeginReadOnlyUntracked()
	{
		// Add a write dep for all arrays to enforce a barrier.
		auto Params = MakeParameters().Write(*this);
		AddProcessingTask(Params, [this]()
		{
		});
		CurrentProcessingStage = EProcessingStage::ReadOnlyUntracked;
		OnProcessingStageChanged();
	}

	/** */
	FParameters MakeParameters();

	bool IsValidIndex(FId Id) const
	{
		return Id.IsValid() && IdAllocator.AllocatedIdMask.IsValidIndex(FDataTraits::IdToIndex(Id)) && IdAllocator.AllocatedIdMask[FDataTraits::IdToIndex(Id)];
	}

	FId Allocate(int32 TypeId)
	{
		return FId(AllocateInternal(TypeId, DirtyState));
	}

	void Free(FId Id)
	{
		check(Id.IsValid());
		// Make sure we're not deleting something that was not marked for delete.
		check(DirtyState.IsDeleted(Id));
		FreeInternal(FDataTraits::IdToIndex(Id), DirtyState);
	}

	struct FFilter
	{
		// TODO: negation, more complex epressions??? Count generalize to predicate, but no way to optimize that (well, can be done but difficult for users to reason about perhaps)

		FFilter& Any(const TBaseDataArray<FDataManager>& Array)
		{
			return *this;
		}

		template <typename DataType>
		FFilter& Any(const TDataArray<DataType, EMapping::Sparse>& Array)
		{
			// Untracked sparse mask: no ChunkSummary available, so TFilteredIdView cannot narrow Any-side.
			AnyItems.Emplace(FItem{ &Array.GetValidMask(), nullptr });
			return *this;
		}

		FFilter& Dirty(const TBaseDataArray<FDataManager>& Array)
		{
			// Only valid in the update stage
			check(Array.DataManager.ResolveAccessContext().ProcessingStage == EProcessingStage::Update);

			const auto& DirtyMasks = Array.DataManager.DirtyState.DirtyMasks;
			if (!DirtyMasks.IsEmpty() && !DirtyMasks[Array.GetRegistrationId()].IsEmpty())
			{
				const FTrackedBitArray& Tracked = DirtyMasks[Array.GetRegistrationId()];
				AnyItems.Emplace(FItem{ &Tracked.Bits, &Tracked.ChunkSummary });
			}
			else
			{
				// Signal that we asked for this but there weren't any
				AnyItems.Emplace(FItem{});
			}
			return *this;
		}

		FFilter& IncludeAdded()
		{
			bIncludeAdded = true;
			return *this;
		}

		FFilter& IncludeDeleted()
		{
			bIncludeDeleted = true;
			return *this;
		}

		FFilter& OnlyDeleted()
		{
			bOnlyDeleted = true;
			return *this;
		}

		// Reset everything
		void IncludeEverything() { *this = FFilter{}; }

	public:
		struct FItem
		{
			const FChunkBitMask* Mask = nullptr;
			const FChunkBitMask* ChunkSummary = nullptr;
		};

		bool IsIncludeAdded() const { return bIncludeAdded; }
		bool IsIncludeDeleted() const { return bIncludeDeleted; }
		bool IsOnlyDeleted() const { return bOnlyDeleted; }
		const TArray<FItem, TInlineAllocator<32>>& GetAllItems() const { return AllItems; }
		const TArray<FItem, TInlineAllocator<8>>& GetAnyItems() const { return AnyItems; }

	protected:
		bool bIncludeAdded = false;
		bool bIncludeDeleted = false;
		bool bOnlyDeleted = false;

		// These flags must all be met (logical AND).
		TArray<FItem, TInlineAllocator<32>> AllItems;
		// One of these must be met (logical OR)
		TArray<FItem, TInlineAllocator<8>> AnyItems;
	};

	struct TFilteredIdView
	{
		const FFilter& Filter;
		const TDataManager& SceneData;

		const FChunkBitMask* AllocatedMask = nullptr;
		// These flags must all be met.
		TArray<const FChunkBitMask*, TInlineAllocator<8>> All;
		// One of these must be met
		TArray<const FChunkBitMask*, TInlineAllocator<8>> Any;
		TChunkBitMask<SceneRenderingAllocator> IterChunkMask;
		bool bIsEmptySet = false;

		bool AreAnyValid() const { return !bIsEmptySet; }

		/**
		 * Upper bound on the number of IDs this view will yield. Counted in units of 64 items.
		 */
		int32 GetUpperBoundCount() const
		{
			if (bIsEmptySet)
			{
				return 0;
			}
			const int32 RawUpper = IterChunkMask.CountSetBits() * FChunkBitMask::BitsPerWord;
			return AllocatedMask ? FMath::Min(RawUpper, AllocatedMask->Num()) : RawUpper;
		}

		TFilteredIdView(const FFilter& InFilter, const TDataManager& InSceneData)
			: Filter(InFilter)
			, SceneData(InSceneData)
		{
			AllocatedMask = &SceneData.IdAllocator.AllocatedIdMask;
			const int32 NumChunks = AllocatedMask->NumChunks();

			// Local-only chunk-summary pointers. Used to compose IterChunkMask without touching
			// the main masks. Live only for the duration of this constructor.
			TArray<const FChunkBitMask*, TInlineAllocator<8>> AllSummaries;
			TArray<const FChunkBitMask*, TInlineAllocator<8>> AnySummaries;

			bool bHasAnyTerm      = false;
			bool bAnyHasUntracked = false;

			auto AddTrackedToAny = [&](const FTrackedBitArray& Tracked)
			{
				Any.Add(&Tracked.Bits);
				AnySummaries.Add(&Tracked.ChunkSummary);
				bHasAnyTerm = true;
			};

			if (Filter.IsIncludeAdded() && !SceneData.DirtyState.AddedMask.IsEmpty())
			{
				check(SceneData.ResolveAccessContext().ProcessingStage == EProcessingStage::Update);
				AddTrackedToAny(SceneData.DirtyState.AddedMask);
			}
			if (Filter.IsIncludeDeleted() && !SceneData.DirtyState.DeletedMask.IsEmpty())
			{
				check(SceneData.ResolveAccessContext().ProcessingStage == EProcessingStage::PreUpdate);
				AddTrackedToAny(SceneData.DirtyState.DeletedMask);
			}
			check(!(Filter.IsOnlyDeleted() && Filter.IsIncludeDeleted()));
			if (Filter.IsOnlyDeleted() && !SceneData.DirtyState.DeletedMask.IsEmpty())
			{
				check(SceneData.ResolveAccessContext().ProcessingStage == EProcessingStage::PreUpdate);
				All.Add(&SceneData.DirtyState.DeletedMask.Bits);
				AllSummaries.Add(&SceneData.DirtyState.DeletedMask.ChunkSummary);
			}

			for (const typename FFilter::FItem& Item : InFilter.GetAnyItems())
			{
				if (Item.Mask)
				{
					Any.Add(Item.Mask);
					bHasAnyTerm = true;
					if (Item.ChunkSummary)
					{
						AnySummaries.Add(Item.ChunkSummary);
					}
					else
					{
						// Untracked term: cannot narrow IterChunkMask against this Any.
						bAnyHasUntracked = true;
					}
				}
			}

			for (const typename FFilter::FItem& Item : InFilter.GetAllItems())
			{
				if (Item.Mask)
				{
					All.Add(Item.Mask);
					if (Item.ChunkSummary)
					{
						AllSummaries.Add(Item.ChunkSummary);
					}
				}
			}

			// None of what was asked for has any changes
			bIsEmptySet = (!bHasAnyTerm && (!Filter.GetAnyItems().IsEmpty() || Filter.IsIncludeAdded() || Filter.IsIncludeDeleted()))
				||		  (All.IsEmpty() && (!Filter.GetAllItems().IsEmpty() || Filter.IsOnlyDeleted()));

			if (!bIsEmptySet && NumChunks > 0)
			{
				// Start IterChunkMask = all chunks in [0, NumChunks) set.
				IterChunkMask.SetNum(NumChunks, false);
				const int32 NumIterWords = IterChunkMask.NumChunks();
				for (int32 WordIndex = 0; WordIndex < NumIterWords; ++WordIndex)
				{
					IterChunkMask.Words[WordIndex] = ~0ULL;
				}
				// Tail-mask the last word so we don't iterate phantom chunks past NumChunks.
				const int32 BitsInLastWord = NumChunks - (NumIterWords - 1) * FChunkBitMask::BitsPerWord;
				if (BitsInLastWord < FChunkBitMask::BitsPerWord)
				{
					IterChunkMask.Words[NumIterWords - 1] = (1ULL << BitsInLastWord) - 1ULL;
				}

				// AND in each All-side summary: term must have a set chunk for that chunk to survive.
				for (const FChunkBitMask* Summary : AllSummaries)
				{
					const int32 SharedWords = FMath::Min(NumIterWords, Summary->NumChunks());
					for (int32 WordIndex = 0; WordIndex < SharedWords; ++WordIndex)
					{
						IterChunkMask.Words[WordIndex] &= Summary->Words[WordIndex];
					}
					// Words past the summary's range contribute zero to an AND.
					for (int32 WordIndex = SharedWords; WordIndex < NumIterWords; ++WordIndex)
					{
						IterChunkMask.Words[WordIndex] = 0ULL;
					}
				}

				// AND in OR-of-Any-summaries. Skip when an untracked Any term is present —
				// we have no summary for it and must not narrow against the OR.
				if (bHasAnyTerm && !bAnyHasUntracked)
				{
					for (int32 WordIndex = 0; WordIndex < NumIterWords; ++WordIndex)
					{
						uint64 AnyWord = 0ULL;
						for (const FChunkBitMask* Summary : AnySummaries)
						{
							AnyWord |= Summary->GetChunkWord(WordIndex);
						}
						IterChunkMask.Words[WordIndex] &= AnyWord;
					}
				}
			}
		}

		struct FIterator
		{
			const TFilteredIdView& View;
			int32  ChunkIndex   = 0;
			uint64 CurrentBits  = 0ULL;
			// Two-level walk state: SummaryChunk is the word index into View.IterChunkMask.
			// SummaryWord holds the *remaining* set bits of that word (cleared as we consume them).
			int32  SummaryChunk = 0;
			uint64 SummaryWord  = 0ULL;

			SCENE_DATA_FORCEINLINE FIterator(const TFilteredIdView& InView, bool bIsTheEnd = false)
				: View(InView)
			{
				const int32 NumSummaryChunks = InView.IterChunkMask.NumChunks();
				if (bIsTheEnd || !InView.AreAnyValid())
				{
					SummaryChunk = NumSummaryChunks;
					SummaryWord  = 0ULL;
					ChunkIndex   = NumSummaryChunks * FChunkBitMask::BitsPerWord;
					CurrentBits  = 0ULL;
				}
				else
				{
					SummaryChunk = 0;
					SummaryWord  = (NumSummaryChunks > 0) ? InView.IterChunkMask.Words[0] : 0ULL;
					ChunkIndex   = 0;
					CurrentBits  = 0ULL;
					AdvanceToNextNonEmptyChunk();
				}
			}

			SCENE_DATA_FORCEINLINE bool operator==(const FIterator& Other) const
			{
				return ChunkIndex == Other.ChunkIndex && CurrentBits == Other.CurrentBits;
			}

			SCENE_DATA_FORCEINLINE bool operator!=(const FIterator& Other) const { return !(*this == Other); }

			SCENE_DATA_FORCEINLINE FId operator*() const
			{
				return FId{ ChunkIndex * FChunkBitMask::BitsPerWord + (int32)FMath::CountTrailingZeros64(CurrentBits) };
			}

			SCENE_DATA_FORCEINLINE explicit operator bool() const { return CurrentBits != 0ULL; }

			SCENE_DATA_FORCEINLINE void operator++()
			{
				CurrentBits &= CurrentBits - 1ULL;
				if (CurrentBits == 0ULL)
				{
					AdvanceToNextNonEmptyChunk();
				}
			}

			SCENE_DATA_FORCEINLINE void AdvanceToNextNonEmptyChunk()
			{
				const int32 NumSummaryChunks = View.IterChunkMask.NumChunks();
				while (true)
				{
					// Walk the chunk-summary until we land on a candidate chunk.
					while (SummaryWord == 0ULL)
					{
						++SummaryChunk;
						if (SummaryChunk >= NumSummaryChunks)
						{
							SummaryChunk = NumSummaryChunks;
							ChunkIndex   = NumSummaryChunks * FChunkBitMask::BitsPerWord;
							CurrentBits  = 0ULL;
							return;
						}
						SummaryWord = View.IterChunkMask.Words[SummaryChunk];
					}

					const int32 BitInSummary = (int32)FMath::CountTrailingZeros64(SummaryWord);
					SummaryWord &= SummaryWord - 1ULL;
					ChunkIndex = SummaryChunk * FChunkBitMask::BitsPerWord + BitInSummary;

					// Inner verification: All-side AND + AllocatedMask + (Any-side OR if present).
					// IterChunkMask proves at least one term in each set is non-zero here, but
					// untracked masks (no summary) and AllocatedMask still need to be sampled.
					uint64 Word = View.AllocatedMask->GetChunkWord(ChunkIndex);
					for (const FChunkBitMask* Mask : View.All)
					{
						Word &= Mask->GetChunkWord(ChunkIndex);
					}
					if (!View.Any.IsEmpty())
					{
						uint64 AnyWord = 0ULL;
						for (const FChunkBitMask* Mask : View.Any)
						{
							AnyWord |= Mask->GetChunkWord(ChunkIndex);
						}
						Word &= AnyWord;
					}
					if (Word != 0ULL)
					{
						CurrentBits = Word;
						return;
					}
				}
			}
		};

		/** Enables range-based for loops, DO NOT USE DIRECTLY. */
		UE_FORCEINLINE_HINT FIterator begin() const { return FIterator(*this); }
		UE_FORCEINLINE_HINT FIterator end()   const { return FIterator(*this, true); }
	};

	FFilter MakeFilter() const { return FFilter{}; }

	TFilteredIdView GetIdView(const FFilter& Filter = FFilter{}) const
	{
		return TFilteredIdView{ Filter, *this };
	}

	void MarkDirty(int32 MemberId, FId ElementId)
	{
		check(ElementId.IsValid());
		DirtyState.MarkDirty(MemberId, IdToIndex(ElementId));
	}

	bool IsDirty(int32 MemberId, FId ElementId) const
	{
		check(ElementId.IsValid());
		return DirtyState.IsDirty(MemberId, IdToIndex(ElementId));
	}

	class FDirtyState : public FBaseDirtyState
	{
	public:
		using FBaseDirtyState::FBaseDirtyState;
		using FBaseDirtyState::Reset;

		bool IsAdded(FId ElementId) const
		{
			check(ElementId.IsValid());
			return AddedMask.IsValidIndex(FDataTraits::IdToIndex(ElementId)) && AddedMask[FDataTraits::IdToIndex(ElementId)];
		}

		bool IsDeleted(FId ElementId) const
		{
			check(ElementId.IsValid());
			return DeletedMask.IsValidIndex(FDataTraits::IdToIndex(ElementId)) && DeletedMask[FDataTraits::IdToIndex(ElementId)];
		}

		void MarkDeleted(FId ElementId)
		{
			check(ElementId.IsValid());
			FBaseDirtyState::MarkDeleted(FDataTraits::IdToIndex(ElementId));
		}
	};
	FDirtyState DirtyState;

	void UnlockAllocations()
	{
		check(bLocked);
		bLocked = false;

		FAccessContext& AccessContext = ResolveAccessContext();

		const uint16 DebugId = SceneData::GetEffectiveDebugId();
		for (FBaseDataArray* Array : DataArrays)
		{
			AccessContext.BeginWriteAccess(Array->GetRegistrationId());
			Array->BeginWrite(DebugId);
		}
	}

	void LockAllocations()
	{
		// TODO: change the bLocked into a stage?
		check(!bLocked);
		bLocked = true;

		FAccessContext& AccessContext = ResolveAccessContext();
		const uint16 DebugId = SceneData::GetEffectiveDebugId();
		for (FBaseDataArray* Array : DataArrays)
		{
			Array->EndWrite(DebugId);
			AccessContext.EndWriteAccess(Array->GetRegistrationId());
		}
		LockAllocationsInternal(DirtyState.DeletedMask.Bits, DirtyState.AddedMask.Bits);
		DirtyState.DirtyMasks.SetNum(DataArrays.Num());
	}

	FAccessContext& GetRootAccessContext() { return DepManager.RootContext; }
	FDepManager& GetDepManager() { return DepManager; }

	SCENE_DATA_FORCEINLINE FAccessContext& ResolveAccessContext() const
	{
		if (const SceneData::FExternalAccessContextContainer* Contexts = SceneData::GetExternalAccessContextsRef())
		{
			if (FAccessContext* AccessContext = Contexts->Get<FAccessContext>())
			{
				return *AccessContext;
			}
		}
		return DepManager.RootContext;
	}
};

template <typename ManagerType>
struct TParameters
{
	explicit TParameters(typename ManagerType::FDepManager& InBuilder, ManagerType& InManager)
		: Builder(InBuilder), Manager(InManager) {}

	TParameters& Read(SceneData::FBaseDataManager& DataManager, EAccessFlags Flags = EAccessFlags::Default)
	{
		for (int32 RegistrationId = 0; RegistrationId < DataManager.GetMaxRegistrationId(); ++RegistrationId)
		{
			ReadDependencies.Emplace(RegistrationId, Flags);
		}
		return *this;
	}

	TParameters& Write(SceneData::FBaseDataManager& DataManager, EAccessFlags Flags = EAccessFlags::Default)
	{
		for (int32 RegistrationId = 0; RegistrationId < DataManager.GetMaxRegistrationId(); ++RegistrationId)
		{
			WriteDependencies.Emplace(RegistrationId, Flags);
		}
		return *this;
	}

	template <typename DataElementType>
	TParameters& Read(const SceneData::TBaseDataArray<DataElementType>& DataArray, EAccessFlags Flags = EAccessFlags::Default)
	{
		ReadDependencies.Emplace(DataArray.GetRegistrationId(), Flags);
		return *this;
	}

	template <typename DataElementType>
	TParameters& Write(SceneData::TBaseDataArray<DataElementType>& DataArray, EAccessFlags Flags = EAccessFlags::Default)
	{
		WriteDependencies.Emplace(DataArray.GetRegistrationId(), Flags);
		return *this;
	}

	typename ManagerType::FDepManager&      Builder;
	ManagerType&                            Manager;
	TArray<FDependencyManager::FParameter>  ReadDependencies;
	TArray<FDependencyManager::FParameter>  WriteDependencies;
};

namespace Detail
{
	/** */
	template <typename ManagerType>
	inline typename ManagerType::FAccessContext* PrepareChildAccessContext(TParameters<ManagerType>& Params, bool bIsRunningAsync, TArray<UE::Tasks::FTask>& Prerequisites, SceneData::FExternalAccessContextContainer& Contexts)
	{
		auto& Dep = Params.Builder;
		Dep.EnsureResourceHeadStateSize(Params.Manager.GetMaxRegistrationId());
		const int32 TaskId = Dep.GetNextTaskId();

		using FAccessContext = typename ManagerType::FAccessContext;
		FAccessContext* AccessContext = Dep.ChildContextStorage.template Create<FAccessContext>(Dep, FBaseAccessContext::FChildTag{});
		AccessContext->ProcessingStage = Params.Manager.GetRootAccessContext().GetProcessingStage();
		Dep.ProcessDependencies(*AccessContext, Prerequisites, TaskId, Params.ReadDependencies, Params.WriteDependencies, bIsRunningAsync);

		Contexts.Set(AccessContext);

		return AccessContext;
	}

	/** */
	template <typename WrapperType>
	inline auto LaunchOrRun(WrapperType&& Wrapper, bool bIsRunningAsync, TArray<UE::Tasks::FTask>&& Prerequisites, const FOptionalTaskArgs& OptArgs)
	{
		using FResult = std::invoke_result_t<WrapperType>;
		UE::Tasks::TTask<FResult> Task;
		if (!bIsRunningAsync)
		{
			UE::Tasks::Wait(Prerequisites);
			if constexpr (std::is_void_v<FResult>)
			{
				Wrapper();
			}
			else
			{
				Task = UE::Tasks::MakeCompletedTask<FResult>(Wrapper());
			}
		}
		else
		{
			// TODO: support launching through pipe if needed.
			check(OptArgs.Pipe == nullptr);
			Task = UE::Tasks::Launch(TEXT("FDependencyManager::Task"), MoveTemp(Wrapper), MoveTemp(Prerequisites), OptArgs.Priority);
		}
		return Task;
	}

	/** */
	template <typename FirstManagerType, typename... RestManagerTypes>
	inline FDependencyManager& PickArenaOwner(TParameters<FirstManagerType>& First, TParameters<RestManagerTypes>&...)
	{
		return First.Builder;
	}

	template <typename TaskLambdaType, typename... ManagerTypes>
	inline auto AddProcessingTaskImpl(FOptionalTaskArgs&& OptArgs, TaskLambdaType&& TaskLambda, TParameters<ManagerTypes>&... ParamsPack)
	{
		using FResult = std::invoke_result_t<TaskLambdaType>;

		TArray<UE::Tasks::FTask> Prerequisites = MoveTemp(OptArgs.Prerequisites);
		bool bIsRunningAsync = ShouldUseParallelRendererTasks() && OptArgs.bCondition;

		FDependencyManager& ArenaOwner = PickArenaOwner(ParamsPack...);
		SceneData::FExternalAccessContextContainer* ArenaContexts = ArenaOwner.ChildContextStorage.template Create<SceneData::FExternalAccessContextContainer>();
		std::tuple<typename ManagerTypes::FAccessContext*...> AccessContextPtrs(PrepareChildAccessContext(ParamsPack , bIsRunningAsync, Prerequisites, *ArenaContexts)...);

		const uint16 TaskDebugId = (uint16)(std::get<0>(AccessContextPtrs)->GetContextId() & 0xFFFFu);

		auto Wrapper = [InnerLambda = MoveTemp(TaskLambda), TaskDebugId]() mutable -> FResult
		{
			FTaskTagScope TaskTagScope(ETaskTag::EParallelRenderingThread);
			SceneData::FDebugIdScope DebugScope(TaskDebugId);
			return InnerLambda();
		};

		FExternalAccessTableScope ContextsScope(*ArenaContexts);
		auto Task = LaunchOrRun(MoveTemp(Wrapper), bIsRunningAsync, MoveTemp(Prerequisites), OptArgs);
		(ParamsPack.Builder.OnTaskLaunched(Task), ...);
		return Task;
	}

	template <typename TaskLambdaType, typename... ManagerTypes>
	inline UE::Tasks::FTask AddSetupTaskImpl(FRDGBuilder& GraphBuilder, FOptionalTaskArgs&& OptArgs, TaskLambdaType&& TaskLambda, TParameters<ManagerTypes>&... ParamsPack)
	{
		TArray<UE::Tasks::FTask> Prerequisites = MoveTemp(OptArgs.Prerequisites);

		FDependencyManager& ArenaOwner = PickArenaOwner(ParamsPack...);
		SceneData::FExternalAccessContextContainer* ArenaContexts =
			ArenaOwner.ChildContextStorage.template Create<SceneData::FExternalAccessContextContainer>();

		std::tuple<typename ManagerTypes::FAccessContext*...> AccessContextPtrs(
			PrepareChildAccessContext(ParamsPack, OptArgs.bCondition, Prerequisites, *ArenaContexts)...);

		const uint16 TaskDebugId = (uint16)(std::get<0>(AccessContextPtrs)->GetContextId() & 0xFFFFu);

		auto Wrapper = [InnerLambda = MoveTemp(TaskLambda), TaskDebugId]() mutable
		{
			SceneData::FDebugIdScope DebugScope(TaskDebugId);
			InnerLambda();
		};

		FExternalAccessTableScope ContextsScope(*ArenaContexts);
		UE::Tasks::FTask Task = GraphBuilder.AddSetupTask(MoveTemp(Wrapper), OptArgs.Pipe, MoveTemp(Prerequisites), OptArgs.Priority, OptArgs.bCondition);
		(ParamsPack.Builder.OnTaskLaunched(Task), ...);
		return Task;
	}
}


template <typename ManagerType, typename TaskLambdaType>
inline auto AddProcessingTask(TParameters<ManagerType>& Params, FOptionalTaskArgs&& OptArgs, TaskLambdaType&& TaskLambda)
{
	return Detail::AddProcessingTaskImpl(MoveTemp(OptArgs), MoveTemp(TaskLambda), Params);
}


template <typename ManagerAType, typename ManagerBType, typename TaskLambdaType>
inline auto AddProcessingTask(TParameters<ManagerAType>& ParamsA, TParameters<ManagerBType>& ParamsB, FOptionalTaskArgs&& OptArgs, TaskLambdaType&& TaskLambda)
{
	return Detail::AddProcessingTaskImpl(MoveTemp(OptArgs), MoveTemp(TaskLambda), ParamsA, ParamsB);
}


template <typename ManagerType, typename TaskLambdaType>
inline UE::Tasks::FTask AddSetupTask(FRDGBuilder& GraphBuilder, TParameters<ManagerType>& Params, FOptionalTaskArgs&& OptArgs, TaskLambdaType&& TaskLambda)
{
	return Detail::AddSetupTaskImpl(GraphBuilder, MoveTemp(OptArgs), MoveTemp(TaskLambda), Params);
}


template <typename ManagerAType, typename ManagerBType, typename TaskLambdaType>
inline UE::Tasks::FTask AddSetupTask(FRDGBuilder& GraphBuilder, TParameters<ManagerAType>& ParamsA, TParameters<ManagerBType>& ParamsB, FOptionalTaskArgs&& OptArgs, TaskLambdaType&& TaskLambda)
{
	return Detail::AddSetupTaskImpl(GraphBuilder, MoveTemp(OptArgs), MoveTemp(TaskLambda), ParamsA, ParamsB);
}


template <typename ManagerType, typename TaskLambdaType>
inline auto AddProcessingTask(TParameters<ManagerType>& Params, TaskLambdaType&& TaskLambda)
{
	return AddProcessingTask(Params, FOptionalTaskArgs{}, MoveTemp(TaskLambda));
}

template <typename ManagerType, typename TaskLambdaType>
inline UE::Tasks::FTask AddSetupTask(FRDGBuilder& GraphBuilder, TParameters<ManagerType>& Params, TaskLambdaType&& TaskLambda)
{
	return AddSetupTask(GraphBuilder, Params, FOptionalTaskArgs{}, MoveTemp(TaskLambda));
}

template <typename ManagerAType, typename ManagerBType, typename TaskLambdaType>
inline auto AddProcessingTask(TParameters<ManagerAType>& ParamsA, TParameters<ManagerBType>& ParamsB, TaskLambdaType&& TaskLambda)
{
	return AddProcessingTask(ParamsA, ParamsB, FOptionalTaskArgs{}, MoveTemp(TaskLambda));
}

template <typename ManagerAType, typename ManagerBType, typename TaskLambdaType>
inline UE::Tasks::FTask AddSetupTask(FRDGBuilder& GraphBuilder, TParameters<ManagerAType>& ParamsA, TParameters<ManagerBType>& ParamsB, TaskLambdaType&& TaskLambda)
{
	return AddSetupTask(GraphBuilder, ParamsA, ParamsB, FOptionalTaskArgs{}, MoveTemp(TaskLambda));
}

template <typename InDataTraitsType>
inline TParameters<TDataManager<InDataTraitsType>> TDataManager<InDataTraitsType>::MakeParameters()
{
	return TParameters<TDataManager>{DepManager, *this};
}

template <typename... ManagerTypes>
inline void Wait(TParameters<ManagerTypes>&... ParamsPack)
{
	Detail::AddProcessingTaskImpl({ .bCondition = false }, [](){}, ParamsPack...);
}

};
