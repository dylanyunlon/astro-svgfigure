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

// [ASTRO-DISPATCH] Z-layer routing dispatcher: messages are dispatched to
// registered ChannelBuffers via a z-layer priority map keyed on channel_id.
// Each Dispatch() call resolves the target z-layer, iterates subscribers in
// layer order, and fills their buffers. Debug prefix [ASTRO-DISPATCH] marks
// all diagnostic paths in this unit.

#ifndef CYBER_DATA_DATA_DISPATCHER_H_
#define CYBER_DATA_DATA_DISPATCHER_H_

#include <memory>
#include <mutex>
#include <vector>

#include "cyber/common/log.h"
#include "cyber/common/macros.h"
#include "cyber/data/channel_buffer.h"
#include "cyber/state.h"
#include "cyber/time/time.h"

namespace apollo {
namespace cyber {
namespace data {

using apollo::cyber::Time;
using apollo::cyber::base::AtomicHashMap;

// ---------------------------------------------------------------------------
// DataDispatcher — z-layer routing dispatcher
//
// Maintains a channel_id → BufferVector map (the "z-layer table").  When
// Dispatch() is called the dispatcher looks up the z-layer entry for the
// given channel_id, then fills every live subscriber buffer in registration
// order (z-order).  Stale weak_ptr entries (dropped subscribers) are silently
// skipped; only live locks proceed.
// ---------------------------------------------------------------------------
template <typename T>
class DataDispatcher {
 public:
  using BufferVector =
      std::vector<std::weak_ptr<CacheBuffer<std::shared_ptr<T>>>>;
  ~DataDispatcher() {}

  // AddBuffer registers a ChannelBuffer into the z-layer routing table.
  // If the channel_id z-layer already exists the buffer is appended (back of
  // z-order); otherwise a new z-layer entry is created.
  void AddBuffer(const ChannelBuffer<T>& channel_buffer);

  // Dispatch routes msg to all z-layer subscribers registered under
  // channel_id.  Returns false if the runtime is shutting down or no z-layer
  // entry exists for channel_id.
  bool Dispatch(const uint64_t channel_id, const std::shared_ptr<T>& msg);

 private:
  DataNotifier* notifier_ = DataNotifier::Instance();
  std::mutex buffers_map_mutex_;
  AtomicHashMap<uint64_t, BufferVector> buffers_map_;

  DECLARE_SINGLETON(DataDispatcher)
};

template <typename T>
inline DataDispatcher<T>::DataDispatcher() {
  ADEBUG << "[ASTRO-DISPATCH] DataDispatcher singleton constructed";
}

template <typename T>
void DataDispatcher<T>::AddBuffer(const ChannelBuffer<T>& channel_buffer) {
  std::lock_guard<std::mutex> lock(buffers_map_mutex_);
  auto buffer = channel_buffer.Buffer();
  BufferVector* buffers = nullptr;
  if (buffers_map_.Get(channel_buffer.channel_id(), &buffers)) {
    buffers->emplace_back(buffer);
    ADEBUG << "[ASTRO-DISPATCH] AddBuffer: appended to existing z-layer "
           << "channel_id=" << channel_buffer.channel_id()
           << " z-depth=" << buffers->size();
  } else {
    BufferVector new_buffers = {buffer};
    buffers_map_.Set(channel_buffer.channel_id(), new_buffers);
    ADEBUG << "[ASTRO-DISPATCH] AddBuffer: created new z-layer entry "
           << "channel_id=" << channel_buffer.channel_id();
  }
}

template <typename T>
bool DataDispatcher<T>::Dispatch(const uint64_t channel_id,
                                 const std::shared_ptr<T>& msg) {
  ADEBUG << "[ASTRO-DISPATCH] Dispatch: z-layer routing for channel_id="
         << channel_id;

  if (apollo::cyber::IsShutdown()) {
    ADEBUG << "[ASTRO-DISPATCH] Dispatch: runtime shutdown, dropping message "
           << "on channel_id=" << channel_id;
    return false;
  }

  BufferVector* buffers = nullptr;
  if (buffers_map_.Get(channel_id, &buffers)) {
    ADEBUG << "[ASTRO-DISPATCH] Dispatch: z-layer found, subscriber count="
           << buffers->size() << " channel_id=" << channel_id;
    uint32_t filled = 0;
    uint32_t skipped = 0;
    for (auto& buffer_wptr : *buffers) {
      if (auto buffer = buffer_wptr.lock()) {
        std::lock_guard<std::mutex> lock(buffer->Mutex());
        buffer->Fill(msg);
        ++filled;
      } else {
        // Subscriber has been destroyed; z-layer entry is stale.
        ++skipped;
        ADEBUG << "[ASTRO-DISPATCH] Dispatch: stale z-layer subscriber "
               << "skipped on channel_id=" << channel_id;
      }
    }
    ADEBUG << "[ASTRO-DISPATCH] Dispatch: z-layer fill done "
           << "filled=" << filled << " skipped=" << skipped
           << " channel_id=" << channel_id;
  } else {
    ADEBUG << "[ASTRO-DISPATCH] Dispatch: no z-layer entry for channel_id="
           << channel_id << ", message dropped";
    return false;
  }

  bool notified = notifier_->Notify(channel_id);
  ADEBUG << "[ASTRO-DISPATCH] Dispatch: notifier result=" << notified
         << " channel_id=" << channel_id;
  return notified;
}

}  // namespace data
}  // namespace cyber
}  // namespace apollo

#endif  // CYBER_DATA_DATA_DISPATCHER_H_
