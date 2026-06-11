/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

// [ASTRO-BUFFER] Epoch-constraint channel buffer: wraps CacheBuffer with
// epoch-aware Fetch / Latest / FetchMulti semantics.  Each read operation
// records diagnostic information about buffer state relative to the current
// epoch cursor so that overflow events and epoch transitions are visible in
// debug logs.  Debug prefix [ASTRO-BUFFER] marks all diagnostic paths.

#ifndef CYBER_DATA_CHANNEL_BUFFER_H_
#define CYBER_DATA_CHANNEL_BUFFER_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "cyber/common/global_data.h"
#include "cyber/common/log.h"
#include "cyber/data/data_notifier.h"

namespace apollo {
namespace cyber {
namespace data {

using apollo::cyber::common::GlobalData;

// ---------------------------------------------------------------------------
// ChannelBuffer — epoch-constraint buffer over a shared CacheBuffer
//
// ChannelBuffer provides three read modes:
//   Fetch()       — epoch-indexed sequential read; advances caller's cursor.
//   Latest()      — non-destructive peek at the tail (last filled slot).
//   FetchMulti()  — bulk epoch read of up to fetch_size most-recent entries.
//
// Overflow detection: when the caller's epoch index has fallen behind the
// buffer Head(), a [ASTRO-BUFFER] WARN is emitted and the cursor is snapped
// forward to the current Tail() to resume from the latest epoch boundary.
// ---------------------------------------------------------------------------
template <typename T>
class ChannelBuffer {
 public:
  using BufferType = CacheBuffer<std::shared_ptr<T>>;
  ChannelBuffer(uint64_t channel_id, BufferType* buffer)
      : channel_id_(channel_id), buffer_(buffer) {
    ADEBUG << "[ASTRO-BUFFER] ChannelBuffer ctor: channel_id=" << channel_id_
           << " capacity=" << buffer_->capacity();
  }

  // Fetch performs an epoch-constrained sequential read.
  // *index == 0 → cold start: snap to current Tail (epoch boundary).
  // *index == Tail+1 → epoch is current, no new data yet.
  // *index < Head → epoch overflow: snap forward and log drop count.
  bool Fetch(uint64_t* index, std::shared_ptr<T>& m);  // NOLINT

  // Latest peeks at the buffer tail without advancing any epoch cursor.
  bool Latest(std::shared_ptr<T>& m);  // NOLINT

  // FetchMulti returns up to fetch_size entries anchored at the current epoch
  // tail, oldest-first, for bulk epoch replay.
  bool FetchMulti(uint64_t fetch_size, std::vector<std::shared_ptr<T>>* vec);

  uint64_t channel_id() const { return channel_id_; }
  std::shared_ptr<BufferType> Buffer() const { return buffer_; }

 private:
  uint64_t channel_id_;
  std::shared_ptr<BufferType> buffer_;
};

// ---------------------------------------------------------------------------
// Fetch — epoch-indexed sequential read
// ---------------------------------------------------------------------------
template <typename T>
bool ChannelBuffer<T>::Fetch(uint64_t* index,
                             std::shared_ptr<T>& m) {  // NOLINT
  std::lock_guard<std::mutex> lock(buffer_->Mutex());

  if (buffer_->Empty()) {
    ADEBUG << "[ASTRO-BUFFER] Fetch: buffer empty, channel="
           << GlobalData::GetChannelById(channel_id_);
    return false;
  }

  if (*index == 0) {
    // Cold-start: initialise epoch cursor to current Tail.
    ADEBUG << "[ASTRO-BUFFER] Fetch: cold-start epoch snap to Tail="
           << buffer_->Tail() << " channel="
           << GlobalData::GetChannelById(channel_id_);
    *index = buffer_->Tail();
  } else if (*index == buffer_->Tail() + 1) {
    // Epoch is fully consumed; no new entry available yet.
    ADEBUG << "[ASTRO-BUFFER] Fetch: epoch current, no new data at index="
           << *index << " channel="
           << GlobalData::GetChannelById(channel_id_);
    return false;
  } else if (*index < buffer_->Head()) {
    // Epoch overflow: caller's cursor has been lapped by the write side.
    auto interval = buffer_->Tail() - *index;
    AWARN << "[ASTRO-BUFFER] Fetch: epoch overflow on channel["
          << GlobalData::GetChannelById(channel_id_) << "] "
          << "drop_messages=[" << interval << "] "
          << "stale_epoch_index=[" << *index << "] "
          << "current_epoch_tail=[" << buffer_->Tail() << "] — "
          << "snapping cursor to current epoch boundary";
    *index = buffer_->Tail();
  }

  ADEBUG << "[ASTRO-BUFFER] Fetch: reading epoch index=" << *index
         << " channel=" << GlobalData::GetChannelById(channel_id_);
  m = buffer_->at(*index);
  return true;
}

// ---------------------------------------------------------------------------
// Latest — non-destructive tail peek
// ---------------------------------------------------------------------------
template <typename T>
bool ChannelBuffer<T>::Latest(std::shared_ptr<T>& m) {  // NOLINT
  std::lock_guard<std::mutex> lock(buffer_->Mutex());

  if (buffer_->Empty()) {
    ADEBUG << "[ASTRO-BUFFER] Latest: buffer empty, channel="
           << GlobalData::GetChannelById(channel_id_);
    return false;
  }

  ADEBUG << "[ASTRO-BUFFER] Latest: peeking epoch tail="
         << buffer_->Tail() << " channel="
         << GlobalData::GetChannelById(channel_id_);
  m = buffer_->Back();
  return true;
}

// ---------------------------------------------------------------------------
// FetchMulti — bulk epoch read
// ---------------------------------------------------------------------------
template <typename T>
bool ChannelBuffer<T>::FetchMulti(uint64_t fetch_size,
                                  std::vector<std::shared_ptr<T>>* vec) {
  std::lock_guard<std::mutex> lock(buffer_->Mutex());

  if (buffer_->Empty()) {
    ADEBUG << "[ASTRO-BUFFER] FetchMulti: buffer empty, channel="
           << GlobalData::GetChannelById(channel_id_);
    return false;
  }

  auto num = std::min(buffer_->Size(), fetch_size);
  ADEBUG << "[ASTRO-BUFFER] FetchMulti: epoch bulk read requested="
         << fetch_size << " available=" << buffer_->Size()
         << " reading=" << num << " channel="
         << GlobalData::GetChannelById(channel_id_);
  vec->reserve(num);
  for (auto index = buffer_->Tail() - num + 1; index <= buffer_->Tail();
       ++index) {
    vec->emplace_back(buffer_->at(index));
  }
  ADEBUG << "[ASTRO-BUFFER] FetchMulti: epoch bulk read complete count="
         << vec->size() << " channel="
         << GlobalData::GetChannelById(channel_id_);
  return true;
}

}  // namespace data
}  // namespace cyber
}  // namespace apollo

#endif  // CYBER_DATA_CHANNEL_BUFFER_H_
