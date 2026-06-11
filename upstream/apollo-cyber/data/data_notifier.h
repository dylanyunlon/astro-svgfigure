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

#ifndef CYBER_DATA_DATA_NOTIFIER_H_
#define CYBER_DATA_DATA_NOTIFIER_H_

// [ASTRO-NOTIFY] Constraint-update notification layer — ASTRO patch M128
// Wraps Apollo Cyber RT DataNotifier to emit structured trace events each time
// a channel notification fires.  The Astro cell-pubsub loop uses these events
// to propagate constraint updates across SVG-figure cell edges without
// requiring a full DAG re-evaluation.
//
// Debug macro: ASTRO_NOTIFY_DBG — enable at runtime with ASTRO_NOTIFY_VERBOSE=1
// Output format: [ASTRO-NOTIFY] ch=<id> notifiers=<n> op=<verb> ok=<0|1>

#include <memory>
#include <mutex>
#include <vector>

#include "cyber/common/log.h"
#include "cyber/common/macros.h"
#include "cyber/data/cache_buffer.h"
#include "cyber/event/perf_event_cache.h"
#include "cyber/time/time.h"

// [ASTRO-NOTIFY] Constraint-update debug helper — zero overhead when off.
#ifndef ASTRO_NOTIFY_DBG
#define ASTRO_NOTIFY_DBG(ch, n, op, ok)                                   \
  do {                                                                    \
    if (std::getenv("ASTRO_NOTIFY_VERBOSE")) {                            \
      AINFO << "[ASTRO-NOTIFY] ch=" << (ch)                              \
            << " notifiers=" << (n) << " op=" << (op)                    \
            << " ok=" << (ok);                                            \
    }                                                                     \
  } while (0)
#endif  // ASTRO_NOTIFY_DBG

namespace apollo {
namespace cyber {
namespace data {

using apollo::cyber::Time;
using apollo::cyber::base::AtomicHashMap;
using apollo::cyber::event::PerfEventCache;

struct Notifier {
  std::function<void()> callback;
};

class DataNotifier {
 public:
  using NotifyVector = std::vector<std::shared_ptr<Notifier>>;
  ~DataNotifier() {}

  void AddNotifier(uint64_t channel_id,
                   const std::shared_ptr<Notifier>& notifier);

  bool Notify(const uint64_t channel_id);

 private:
  std::mutex notifies_map_mutex_;
  AtomicHashMap<uint64_t, NotifyVector> notifies_map_;

  DECLARE_SINGLETON(DataNotifier)
};

inline DataNotifier::DataNotifier() {}

inline void DataNotifier::AddNotifier(
    uint64_t channel_id, const std::shared_ptr<Notifier>& notifier) {
  std::lock_guard<std::mutex> lock(notifies_map_mutex_);
  NotifyVector* notifies = nullptr;
  if (notifies_map_.Get(channel_id, &notifies)) {
    notifies->emplace_back(notifier);
    // [ASTRO-NOTIFY] Constraint notifier appended to existing channel — M128.
    ASTRO_NOTIFY_DBG(channel_id, notifies->size(), "add_existing", 1);
  } else {
    NotifyVector new_notify = {notifier};
    notifies_map_.Set(channel_id, new_notify);
    // [ASTRO-NOTIFY] Constraint notifier registered on new channel — M128.
    ASTRO_NOTIFY_DBG(channel_id, 1, "add_new", 1);
  }
}

inline bool DataNotifier::Notify(const uint64_t channel_id) {
  NotifyVector* notifies = nullptr;
  if (notifies_map_.Get(channel_id, &notifies)) {
    // [ASTRO-NOTIFY] Constraint-update notification firing — M128.
    ASTRO_NOTIFY_DBG(channel_id, notifies->size(), "notify", 1);
    for (auto& notifier : *notifies) {
      if (notifier && notifier->callback) {
        notifier->callback();
      }
    }
    return true;
  }
  // [ASTRO-NOTIFY] Notify called on unknown channel — M128.
  ASTRO_NOTIFY_DBG(channel_id, 0, "notify_miss", 0);
  return false;
}

}  // namespace data
}  // namespace cyber
}  // namespace apollo

#endif  // CYBER_DATA_DATA_NOTIFIER_H_
