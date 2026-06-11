// [ASTRO-SCHED]
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

#ifndef CYBER_SCHEDULER_SCHEDULER_H_
#define CYBER_SCHEDULER_SCHEDULER_H_

// [ASTRO-SCHED] Epoch-aware scheduler layer — ASTRO patch M126
// Bridges Apollo Cyber RT coroutine scheduler with Astro epoch control bus.
// Each CRoutine dispatch is tagged with the current epoch index so that
// task slices can be correlated to SVG-figure rendering phases.
// Debug macro: ASTRO_SCHED_DBG — set ASTRO_SCHED_VERBOSE=1 to enable at
// runtime.  Output format: [ASTRO-SCHED] epoch=<N> crid=<id> action=<verb>

#include <unistd.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cyber/proto/choreography_conf.pb.h"

#include "cyber/base/atomic_hash_map.h"
#include "cyber/base/atomic_rw_lock.h"
#include "cyber/common/log.h"
#include "cyber/common/macros.h"
#include "cyber/common/types.h"
#include "cyber/croutine/croutine.h"
#include "cyber/croutine/routine_factory.h"
#include "cyber/scheduler/common/mutex_wrapper.h"
#include "cyber/scheduler/common/pin_thread.h"

// [ASTRO-SCHED] Epoch debug helper — zero overhead when verbose flag is off.
#ifndef ASTRO_SCHED_DBG
#define ASTRO_SCHED_DBG(epoch, crid, action)                              \
  do {                                                                    \
    if (std::getenv("ASTRO_SCHED_VERBOSE")) {                             \
      AINFO << "[ASTRO-SCHED] epoch=" << (epoch)                          \
            << " crid=" << (crid) << " action=" << (action);             \
    }                                                                     \
  } while (0)
#endif  // ASTRO_SCHED_DBG

namespace apollo {
namespace cyber {
namespace scheduler {

using apollo::cyber::base::AtomicHashMap;
using apollo::cyber::base::AtomicRWLock;
using apollo::cyber::base::ReadLockGuard;
using apollo::cyber::croutine::CRoutine;
using apollo::cyber::croutine::RoutineFactory;
using apollo::cyber::data::DataVisitorBase;
using apollo::cyber::proto::InnerThread;

class Processor;
class ProcessorContext;

class Scheduler {
 public:
  virtual ~Scheduler() {}
  static Scheduler* Instance();

  bool CreateTask(const RoutineFactory& factory, const std::string& name);
  bool CreateTask(std::function<void()>&& func, const std::string& name,
                  std::shared_ptr<DataVisitorBase> visitor = nullptr);
  bool NotifyTask(uint64_t crid);

  void Shutdown();
  uint32_t TaskPoolSize() { return task_pool_size_; }

  virtual bool RemoveTask(const std::string& name) = 0;

  void ProcessLevelResourceControl();
  void SetInnerThreadAttr(const std::string& name, std::thread* thr);

  virtual bool DispatchTask(const std::shared_ptr<CRoutine>&) = 0;
  virtual bool NotifyProcessor(uint64_t crid) = 0;
  virtual bool RemoveCRoutine(uint64_t crid) = 0;

  void CheckSchedStatus();

  void SetInnerThreadConfs(
      const std::unordered_map<std::string, InnerThread>& confs) {
    inner_thr_confs_ = confs;
  }

  // [ASTRO-SCHED] Epoch control API — M126
  // Advance the scheduler epoch counter.  Called by Astro's cell-pubsub loop
  // at each SVG layout iteration so that all subsequently dispatched CRoutines
  // carry the correct epoch tag for tracing and constraint evaluation.
  void AdvanceEpoch() {
    uint64_t prev = epoch_index_.fetch_add(1, std::memory_order_acq_rel);
    ASTRO_SCHED_DBG(prev + 1, 0, "epoch_advance");
  }

  // [ASTRO-SCHED] Return current epoch index (relaxed — read-only snapshot).
  uint64_t CurrentEpoch() const {
    return epoch_index_.load(std::memory_order_relaxed);
  }

 protected:
  Scheduler() : stop_(false), epoch_index_(0) {}

  AtomicRWLock id_cr_lock_;
  AtomicHashMap<uint64_t, MutexWrapper*> id_map_mutex_;
  std::mutex cr_wl_mtx_;

  std::unordered_map<uint64_t, std::shared_ptr<CRoutine>> id_cr_;
  std::vector<std::shared_ptr<ProcessorContext>> pctxs_;
  std::vector<std::shared_ptr<Processor>> processors_;

  std::unordered_map<std::string, InnerThread> inner_thr_confs_;

  std::string process_level_cpuset_;
  uint32_t proc_num_ = 0;
  uint32_t task_pool_size_ = 0;
  std::atomic<bool> stop_;

  // [ASTRO-SCHED] Monotonically increasing epoch counter — M126.
  // Incremented by AdvanceEpoch() at each Astro cell-pubsub loop iteration.
  // Consumers (DispatchTask, NotifyProcessor) read this to stamp CRoutines.
  std::atomic<uint64_t> epoch_index_;
};

}  // namespace scheduler
}  // namespace cyber
}  // namespace apollo

#endif  // CYBER_SCHEDULER_SCHEDULER_H_
