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
// [ASTRO-PROCESSOR] Repurposed: CRoutine processor thread → epoch-slot executor.
// Each Processor::Run loop iteration = one epoch tick; croutine->Resume() advances
// a single cell through its epoch computation. context_->Wait() = inter-epoch idle.

#include "cyber/scheduler/processor.h"

#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>

#include <chrono>

#include "cyber/common/global_data.h"
#include "cyber/common/log.h"
#include "cyber/croutine/croutine.h"
#include "cyber/time/time.h"

namespace apollo {
namespace cyber {
namespace scheduler {

using apollo::cyber::common::GlobalData;

Processor::Processor() { running_.store(true); }

Processor::~Processor() { Stop(); }

void Processor::Run() {
  tid_.store(static_cast<int>(syscall(SYS_gettid)));
  AINFO << "processor_tid: " << tid_;
  snap_shot_->processor_id.store(tid_);

  // [ASTRO-PROCESSOR] Epoch executor thread online.
  // Each loop iteration = one epoch tick; croutine resume = cell epoch computation.
  fprintf(stderr, "[ASTRO-PROCESSOR] Processor::Run: epoch executor tid=%d ONLINE\n",
      static_cast<int>(tid_.load()));

  while (cyber_likely(running_.load())) {
    if (cyber_likely(context_ != nullptr)) {
      auto croutine = context_->NextRoutine();
      if (croutine) {
        snap_shot_->execute_start_time.store(cyber::Time::Now().ToNanosecond());
        snap_shot_->routine_name = croutine->name();
        // [ASTRO-PROCESSOR] Epoch tick: advancing cell through epoch computation.
        fprintf(stderr, "[ASTRO-PROCESSOR] epoch tick: cell='%s' tid=%d\n",
            croutine->name().c_str(), static_cast<int>(tid_.load()));
        croutine->Resume();
        croutine->Release();
      } else {
        snap_shot_->execute_start_time.store(0);
        context_->Wait();
      }
    } else {
      std::unique_lock<std::mutex> lk(mtx_ctx_);
      cv_ctx_.wait_for(lk, std::chrono::milliseconds(10));
    }
  }
}

void Processor::Stop() {
  if (!running_.exchange(false)) {
    return;
  }

  if (context_) {
    context_->Shutdown();
  }

  cv_ctx_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Processor::BindContext(const std::shared_ptr<ProcessorContext>& context) {
  context_ = context;
  std::call_once(thread_flag_,
                 [this]() { thread_ = std::thread(&Processor::Run, this); });
}

std::atomic<pid_t>& Processor::Tid() {
  while (tid_.load() == -1) {
    cpu_relax();
  }
  return tid_;
}

}  // namespace scheduler
}  // namespace cyber
}  // namespace apollo
