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

// [ASTRO-VISITOR] ConstraintVisitor: constraint-aware traversal over fused
// channel buffers. Each specialization registers its ChannelBuffers with the
// DataDispatcher and drives a DataFusion pass under index-constraint semantics.
// Debug prefix [ASTRO-VISITOR] marks all diagnostic paths in this unit.

#ifndef CYBER_DATA_DATA_VISITOR_H_
#define CYBER_DATA_DATA_VISITOR_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "cyber/common/log.h"
#include "cyber/data/channel_buffer.h"
#include "cyber/data/data_dispatcher.h"
#include "cyber/data/data_visitor_base.h"
#include "cyber/data/fusion/all_latest.h"
#include "cyber/data/fusion/data_fusion.h"

namespace apollo {
namespace cyber {
namespace data {

// VisitorConfig carries per-channel constraint parameters used by
// ConstraintVisitor to bind a ChannelBuffer at construction time.
struct VisitorConfig {
  VisitorConfig(uint64_t id, uint32_t size)
      : channel_id(id), queue_size(size) {}
  uint64_t channel_id;
  uint32_t queue_size;
};

template <typename T>
using BufferType = CacheBuffer<std::shared_ptr<T>>;

// ---------------------------------------------------------------------------
// ConstraintVisitor<M0,M1,M2,M3> — 4-channel constraint traversal
// ---------------------------------------------------------------------------
template <typename M0, typename M1 = NullType, typename M2 = NullType,
          typename M3 = NullType>
class DataVisitor : public DataVisitorBase {
 public:
  explicit DataVisitor(const std::vector<VisitorConfig>& configs)
      : buffer_m0_(configs[0].channel_id,
                   new BufferType<M0>(configs[0].queue_size)),
        buffer_m1_(configs[1].channel_id,
                   new BufferType<M1>(configs[1].queue_size)),
        buffer_m2_(configs[2].channel_id,
                   new BufferType<M2>(configs[2].queue_size)),
        buffer_m3_(configs[3].channel_id,
                   new BufferType<M3>(configs[3].queue_size)) {
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0,M1,M2,M3> ctor: "
           << "binding 4 channel-buffers ch0=" << configs[0].channel_id
           << " ch1=" << configs[1].channel_id
           << " ch2=" << configs[2].channel_id
           << " ch3=" << configs[3].channel_id;
    DataDispatcher<M0>::Instance()->AddBuffer(buffer_m0_);
    DataDispatcher<M1>::Instance()->AddBuffer(buffer_m1_);
    DataDispatcher<M2>::Instance()->AddBuffer(buffer_m2_);
    DataDispatcher<M3>::Instance()->AddBuffer(buffer_m3_);
    data_notifier_->AddNotifier(buffer_m0_.channel_id(), notifier_);
    data_fusion_ = new fusion::AllLatest<M0, M1, M2, M3>(
        buffer_m0_, buffer_m1_, buffer_m2_, buffer_m3_);
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0,M1,M2,M3> fusion ready, "
           << "notifier registered on ch0=" << configs[0].channel_id;
  }

  ~DataVisitor() {
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0,M1,M2,M3> dtor: "
           << "releasing fusion at next_msg_index_=" << next_msg_index_;
    if (data_fusion_) {
      delete data_fusion_;
      data_fusion_ = nullptr;
    }
  }

  // TryFetch performs a constraint-gated fusion pass.
  // Returns true only when all four channel constraints are simultaneously
  // satisfiable at next_msg_index_; advances the constraint cursor on success.
  bool TryFetch(std::shared_ptr<M0>& m0, std::shared_ptr<M1>& m1,    // NOLINT
                std::shared_ptr<M2>& m2, std::shared_ptr<M3>& m3) {  // NOLINT
    ADEBUG << "[ASTRO-VISITOR] TryFetch<4> at constraint index="
           << next_msg_index_;
    if (data_fusion_->Fusion(&next_msg_index_, m0, m1, m2, m3)) {
      ADEBUG << "[ASTRO-VISITOR] TryFetch<4> constraint satisfied, "
             << "advancing index to " << (next_msg_index_ + 1);
      next_msg_index_++;
      return true;
    }
    ADEBUG << "[ASTRO-VISITOR] TryFetch<4> constraint not met at index="
           << next_msg_index_;
    return false;
  }

 private:
  fusion::DataFusion<M0, M1, M2, M3>* data_fusion_ = nullptr;
  ChannelBuffer<M0> buffer_m0_;
  ChannelBuffer<M1> buffer_m1_;
  ChannelBuffer<M2> buffer_m2_;
  ChannelBuffer<M3> buffer_m3_;
};

// ---------------------------------------------------------------------------
// ConstraintVisitor<M0,M1,M2> — 3-channel constraint traversal
// ---------------------------------------------------------------------------
template <typename M0, typename M1, typename M2>
class DataVisitor<M0, M1, M2, NullType> : public DataVisitorBase {
 public:
  explicit DataVisitor(const std::vector<VisitorConfig>& configs)
      : buffer_m0_(configs[0].channel_id,
                   new BufferType<M0>(configs[0].queue_size)),
        buffer_m1_(configs[1].channel_id,
                   new BufferType<M1>(configs[1].queue_size)),
        buffer_m2_(configs[2].channel_id,
                   new BufferType<M2>(configs[2].queue_size)) {
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0,M1,M2> ctor: "
           << "binding 3 channel-buffers ch0=" << configs[0].channel_id
           << " ch1=" << configs[1].channel_id
           << " ch2=" << configs[2].channel_id;
    DataDispatcher<M0>::Instance()->AddBuffer(buffer_m0_);
    DataDispatcher<M1>::Instance()->AddBuffer(buffer_m1_);
    DataDispatcher<M2>::Instance()->AddBuffer(buffer_m2_);
    data_notifier_->AddNotifier(buffer_m0_.channel_id(), notifier_);
    data_fusion_ =
        new fusion::AllLatest<M0, M1, M2>(buffer_m0_, buffer_m1_, buffer_m2_);
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0,M1,M2> fusion ready";
  }

  ~DataVisitor() {
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0,M1,M2> dtor";
    if (data_fusion_) {
      delete data_fusion_;
      data_fusion_ = nullptr;
    }
  }

  bool TryFetch(std::shared_ptr<M0>& m0, std::shared_ptr<M1>& m1,  // NOLINT
                std::shared_ptr<M2>& m2) {                         // NOLINT
    ADEBUG << "[ASTRO-VISITOR] TryFetch<3> at constraint index="
           << next_msg_index_;
    if (data_fusion_->Fusion(&next_msg_index_, m0, m1, m2)) {
      ADEBUG << "[ASTRO-VISITOR] TryFetch<3> constraint satisfied, "
             << "advancing index to " << (next_msg_index_ + 1);
      next_msg_index_++;
      return true;
    }
    ADEBUG << "[ASTRO-VISITOR] TryFetch<3> constraint not met at index="
           << next_msg_index_;
    return false;
  }

 private:
  fusion::DataFusion<M0, M1, M2>* data_fusion_ = nullptr;
  ChannelBuffer<M0> buffer_m0_;
  ChannelBuffer<M1> buffer_m1_;
  ChannelBuffer<M2> buffer_m2_;
};

// ---------------------------------------------------------------------------
// ConstraintVisitor<M0,M1> — 2-channel constraint traversal
// ---------------------------------------------------------------------------
template <typename M0, typename M1>
class DataVisitor<M0, M1, NullType, NullType> : public DataVisitorBase {
 public:
  explicit DataVisitor(const std::vector<VisitorConfig>& configs)
      : buffer_m0_(configs[0].channel_id,
                   new BufferType<M0>(configs[0].queue_size)),
        buffer_m1_(configs[1].channel_id,
                   new BufferType<M1>(configs[1].queue_size)) {
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0,M1> ctor: "
           << "binding 2 channel-buffers ch0=" << configs[0].channel_id
           << " ch1=" << configs[1].channel_id;
    DataDispatcher<M0>::Instance()->AddBuffer(buffer_m0_);
    DataDispatcher<M1>::Instance()->AddBuffer(buffer_m1_);
    data_notifier_->AddNotifier(buffer_m0_.channel_id(), notifier_);
    data_fusion_ = new fusion::AllLatest<M0, M1>(buffer_m0_, buffer_m1_);
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0,M1> fusion ready";
  }

  ~DataVisitor() {
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0,M1> dtor";
    if (data_fusion_) {
      delete data_fusion_;
      data_fusion_ = nullptr;
    }
  }

  bool TryFetch(std::shared_ptr<M0>& m0, std::shared_ptr<M1>& m1) {  // NOLINT
    ADEBUG << "[ASTRO-VISITOR] TryFetch<2> at constraint index="
           << next_msg_index_;
    if (data_fusion_->Fusion(&next_msg_index_, m0, m1)) {
      ADEBUG << "[ASTRO-VISITOR] TryFetch<2> constraint satisfied, "
             << "advancing index to " << (next_msg_index_ + 1);
      next_msg_index_++;
      return true;
    }
    ADEBUG << "[ASTRO-VISITOR] TryFetch<2> constraint not met at index="
           << next_msg_index_;
    return false;
  }

 private:
  fusion::DataFusion<M0, M1>* data_fusion_ = nullptr;
  ChannelBuffer<M0> buffer_m0_;
  ChannelBuffer<M1> buffer_m1_;
};

// ---------------------------------------------------------------------------
// ConstraintVisitor<M0> — single-channel constraint traversal
// ---------------------------------------------------------------------------
template <typename M0>
class DataVisitor<M0, NullType, NullType, NullType> : public DataVisitorBase {
 public:
  explicit DataVisitor(const VisitorConfig& configs)
      : buffer_(configs.channel_id, new BufferType<M0>(configs.queue_size)) {
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0> ctor (VisitorConfig): "
           << "channel_id=" << configs.channel_id
           << " queue_size=" << configs.queue_size;
    DataDispatcher<M0>::Instance()->AddBuffer(buffer_);
    data_notifier_->AddNotifier(buffer_.channel_id(), notifier_);
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0> notifier registered";
  }

  DataVisitor(uint64_t channel_id, uint32_t queue_size)
      : buffer_(channel_id, new BufferType<M0>(queue_size)) {
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0> ctor (explicit): "
           << "channel_id=" << channel_id
           << " queue_size=" << queue_size;
    DataDispatcher<M0>::Instance()->AddBuffer(buffer_);
    data_notifier_->AddNotifier(buffer_.channel_id(), notifier_);
    ADEBUG << "[ASTRO-VISITOR] ConstraintVisitor<M0> notifier registered";
  }

  // TryFetch advances the single-channel constraint cursor.
  bool TryFetch(std::shared_ptr<M0>& m0) {  // NOLINT
    ADEBUG << "[ASTRO-VISITOR] TryFetch<1> at constraint index="
           << next_msg_index_;
    if (buffer_.Fetch(&next_msg_index_, m0)) {
      ADEBUG << "[ASTRO-VISITOR] TryFetch<1> fetch ok, advancing index to "
             << (next_msg_index_ + 1);
      next_msg_index_++;
      return true;
    }
    ADEBUG << "[ASTRO-VISITOR] TryFetch<1> buffer empty or constraint not met "
           << "at index=" << next_msg_index_;
    return false;
  }

 private:
  ChannelBuffer<M0> buffer_;
};

}  // namespace data
}  // namespace cyber
}  // namespace apollo

#endif  // CYBER_DATA_DATA_VISITOR_H_
