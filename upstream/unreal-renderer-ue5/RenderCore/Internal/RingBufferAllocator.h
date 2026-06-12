// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreTypes.h"
#include "Containers/Queue.h"
#include "Misc/AssertionMacros.h"

class FRingBufferAllocator
{
public:
	FRingBufferAllocator(uint32 Size):
		BufferSize(Size)
	{
		Reset();
	}

	void Reset()
	{
		ReadOffset = 0u;
		WriteOffset = 0u;
	#if DO_CHECK
		SizeQueue.Empty();
	#endif
	}

	bool TryAllocate(uint32 Size, uint32& AllocatedOffset)
	{
		if (WriteOffset < ReadOffset)
		{
			if (Size + 1u > ReadOffset - WriteOffset)	// +1 to leave one element free, so we can distinguish between full and empty
			{
				return false;
			}
		}
		else
		{
			// WriteOffset >= ReadOffset
			if (Size + (ReadOffset == 0u ? 1u : 0u) > BufferSize - WriteOffset)
			{
				// Doesn't fit at the end. Try from the beginning
				if (Size + 1u > ReadOffset)
				{
					return false;
				}
				WriteOffset = 0u;
			}
		}

	#if DO_CHECK
		SizeQueue.Enqueue(Size);
	#endif
		AllocatedOffset = WriteOffset;
		WriteOffset += Size;
		check(AllocatedOffset + Size <= BufferSize);
		return true;
	}

	void Free(uint32 Size)
	{
	#if DO_CHECK
		uint32 QueuedSize;
		bool bNonEmpty = SizeQueue.Dequeue(QueuedSize);
		check(bNonEmpty);
		check(QueuedSize == Size);
	#endif
		const uint32 Next = ReadOffset + Size;
		ReadOffset = (Next <= BufferSize) ? Next : Size;
	}
private:
	uint32 BufferSize;
	uint32 ReadOffset;
	uint32 WriteOffset;
#if DO_CHECK
	TQueue<uint32> SizeQueue;
#endif
};
