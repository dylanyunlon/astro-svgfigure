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

#ifndef CYBER_COMPONENT_COMPONENT_H_
#define CYBER_COMPONENT_COMPONENT_H_

#include <memory>
#include <utility>
#include <vector>

// DEBUG: astro-svgfigure pipeline stage marker — 4-channel cell Proc fusion entry
#include <cstdio>

#include "cyber/base/macros.h"
#include "cyber/blocker/blocker_manager.h"
#include "cyber/common/global_data.h"
#include "cyber/common/types.h"
#include "cyber/common/util.h"
#include "cyber/component/component_base.h"
#include "cyber/croutine/routine_factory.h"
#include "cyber/statistics/statistics.h"
#include "cyber/time/time.h"
#include "cyber/data/data_visitor.h"
#include "cyber/scheduler/scheduler.h"

namespace apollo {
namespace cyber {

using apollo::cyber::common::GlobalData;
using apollo::cyber::proto::RoleAttributes;

// =============================================================================
// astro-svgfigure: 4-channel cell Proc fusion — Component pubsub mapping
// =============================================================================
// In Apollo's original Component<M0,M1,M2,M3>, Proc() fuses four independent
// cyber channels into a single coroutine invocation via DataVisitor<M0,M1,M2,M3>.
// In the astro-svgfigure layout pipeline, the four channels map to distinct
// cell signal streams that drive the SVG constraint solver:
//
//   M0 (ch0) → skeleton_signal  : bone/joint topology update from animation rig
//   M1 (ch1) → force_field      : per-cell repulsion/attraction force vectors
//   M2 (ch2) → palette          : color-space assignment for cell fill/stroke
//   M3 (ch3) → z_layers         : stacking order commands (raise/lower/absolute)
//
// When all four channels carry a fresh message, Proc() is invoked once with the
// fused quad-tuple.  The constraint dispatcher then:
//   1. Applies skeleton_signal to update cell anchor positions.
//   2. Integrates force_field to resolve bbox collision (same z-layer push-apart).
//   3. Assigns palette entries to the cell fill/stroke attribute map.
//   4. Commits z_layers to the FAstroZLayerRegistry, triggering re-sort.
//
// astro_component_epoch — incremented on every 4-channel Proc() fusion call.
// astro_proc_fusion_count — total fused dispatches since component Init().
// astro_channel_ready_mask — bitmask of channels with a non-null message this
//                            fusion round (bits 0-3 → ch0-ch3).
// astro_skeleton_seq — last skeleton_signal sequence number seen by Proc().
// astro_zlayer_commit_count — cumulative z_layer commits dispatched so far.
// =============================================================================

/**
 * FAstroCellFusion holds the per-invocation metadata for one 4-channel Proc()
 * call.  It mirrors FProjectedShadowInfo in ShadowRendering in that both act as
 * the central descriptor passed through the dispatch loop; here the "shadow"
 * is the constraint set that the cell must resolve before the next render tick.
 *
 * Fields:
 *   node_name        — cyber node name (identifies the cell sub-Claude instance)
 *   channel_ready    — bitmask: bit i set ↔ channel i delivered a non-null msg
 *   skeleton_seq     — monotonic sequence from the skeleton_signal channel
 *   force_magnitude  — L2-norm of the dominant force vector from force_field ch
 *   palette_id       — palette slot index assigned this fusion round
 *   z_layer_target   — resolved z-layer index after z_layers command applied
 *   proc_latency_us  — microseconds spent inside Proc() (filled after return)
 */
struct FAstroCellFusion {
  const char* node_name;        // cyber node → cell instance identifier
  int32_t     channel_ready;    // bitmask 0b1111 when all 4 channels present
  uint64_t    skeleton_seq;     // skeleton_signal message sequence number
  float       force_magnitude;  // dominant force vector magnitude (force_field)
  int32_t     palette_id;       // palette slot index (palette channel)
  int32_t     z_layer_target;   // committed z-layer after z_layers dispatch
  int64_t     proc_latency_us;  // Proc() wall-clock latency in microseconds

  /** Emit a single-line debug summary to stderr. */
  void DebugPrint() const {
    fprintf(stderr,
            "[ASTRO-COMPONENT] cell-fusion | node=%s ch_mask=0x%x "
            "skel_seq=%lu force=%.3f palette=%d z=%d latency_us=%ld\n",
            node_name, channel_ready, (unsigned long)skeleton_seq,
            force_magnitude, palette_id, z_layer_target, (long)proc_latency_us);
  }
};

// astro-svgfigure: global Component fusion state
// astro_component_epoch is bumped each time the 4-channel DataVisitor fires.
// astro_proc_fusion_count accumulates all successful Proc() dispatches.
// astro_channel_ready_mask records the last fusion round's channel presence bits.
// astro_skeleton_seq tracks the most recent skeleton_signal sequence seen.
// astro_zlayer_commit_count records cumulative z_layer commits.
static int32_t  astro_component_epoch      = 0;
static int32_t  astro_proc_fusion_count    = 0;
static int32_t  astro_channel_ready_mask   = 0;   // bits 0-3 → ch0-ch3
static uint64_t astro_skeleton_seq         = 0;
static int32_t  astro_zlayer_commit_count  = 0;

/**
 * @brief .
 * The Component can process up to four channels of messages. The message type
 * is specified when the component is created. The Component is inherited from
 * ComponentBase. Your component can inherit from Component, and implement
 * Init() & Proc(...), They are picked up by the CyberRT. There are 4
 * specialization implementations.
 *
 * @tparam M0 the first message.
 * @tparam M1 the second message.
 * @tparam M2 the third message.
 * @tparam M3 the fourth message.
 * @warning The Init & Proc functions need to be overloaded, but don't want to
 * be called. They are called by the CyberRT Frame.
 *
 */
template <typename M0 = NullType, typename M1 = NullType,
          typename M2 = NullType, typename M3 = NullType>
class Component : public ComponentBase {
 public:
  Component() {}
  ~Component() override {}

  /**
   * @brief init the component by protobuf object.
   *
   * @param config which is defined in 'cyber/proto/component_conf.proto'
   *
   * @return returns true if successful, otherwise returns false
   */
  bool Initialize(const ComponentConfig& config) override;
  bool Process(const std::shared_ptr<M0>& msg0, const std::shared_ptr<M1>& msg1,
               const std::shared_ptr<M2>& msg2,
               const std::shared_ptr<M3>& msg3);

 private:
  /**
   * @brief The process logical of yours.
   *
   * @param msg0 the first channel message.
   * @param msg1 the second channel message.
   * @param msg2 the third channel message.
   * @param msg3 the fourth channel message.
   *
   * @return returns true if successful, otherwise returns false
   */
  virtual bool Proc(const std::shared_ptr<M0>& msg0,
                    const std::shared_ptr<M1>& msg1,
                    const std::shared_ptr<M2>& msg2,
                    const std::shared_ptr<M3>& msg3) = 0;
};

template <>
class Component<NullType, NullType, NullType, NullType> : public ComponentBase {
 public:
  Component() {}
  ~Component() override {}
  bool Initialize(const ComponentConfig& config) override;
};

template <typename M0>
class Component<M0, NullType, NullType, NullType> : public ComponentBase {
 public:
  Component() {}
  ~Component() override {}
  bool Initialize(const ComponentConfig& config) override;
  bool Process(const std::shared_ptr<M0>& msg);

 private:
  virtual bool Proc(const std::shared_ptr<M0>& msg) = 0;
};

template <typename M0, typename M1>
class Component<M0, M1, NullType, NullType> : public ComponentBase {
 public:
  Component() {}
  ~Component() override {}
  bool Initialize(const ComponentConfig& config) override;
  bool Process(const std::shared_ptr<M0>& msg0,
               const std::shared_ptr<M1>& msg1);

 private:
  virtual bool Proc(const std::shared_ptr<M0>& msg,
                    const std::shared_ptr<M1>& msg1) = 0;
};

template <typename M0, typename M1, typename M2>
class Component<M0, M1, M2, NullType> : public ComponentBase {
 public:
  Component() {}
  ~Component() override {}
  bool Initialize(const ComponentConfig& config) override;
  bool Process(const std::shared_ptr<M0>& msg0, const std::shared_ptr<M1>& msg1,
               const std::shared_ptr<M2>& msg2);

 private:
  virtual bool Proc(const std::shared_ptr<M0>& msg,
                    const std::shared_ptr<M1>& msg1,
                    const std::shared_ptr<M2>& msg2) = 0;
};

template <typename M0>
bool Component<M0, NullType, NullType, NullType>::Process(
    const std::shared_ptr<M0>& msg) {
  if (is_shutdown_.load()) {
    return true;
  }
  return Proc(msg);
}

inline bool Component<NullType, NullType, NullType>::Initialize(
    const ComponentConfig& config) {
  node_.reset(new Node(config.name()));
  LoadConfigFiles(config);
  if (!Init()) {
    AERROR << "Component Init() failed." << std::endl;
    return false;
  }
  return true;
}

template <typename M0>
bool Component<M0, NullType, NullType, NullType>::Initialize(
    const ComponentConfig& config) {
  node_.reset(new Node(config.name()));
  LoadConfigFiles(config);

  if (config.readers_size() < 1) {
    AERROR << "Invalid config file: too few readers.";
    return false;
  }

  if (!Init()) {
    AERROR << "Component Init() failed.";
    return false;
  }

  bool is_reality_mode = GlobalData::Instance()->IsRealityMode();

  ReaderConfig reader_cfg;
  reader_cfg.channel_name = config.readers(0).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(0).qos_profile());
  reader_cfg.pending_queue_size = config.readers(0).pending_queue_size();

  auto role_attr = std::make_shared<proto::RoleAttributes>();
  role_attr->set_node_name(config.name());
  role_attr->set_channel_name(config.readers(0).channel());

  std::weak_ptr<Component<M0>> self =
      std::dynamic_pointer_cast<Component<M0>>(shared_from_this());
  auto func = [self, role_attr](const std::shared_ptr<M0>& msg) {
    auto start_time = Time::Now().ToMicrosecond();
    auto ptr = self.lock();
    if (ptr) {
      ptr->Process(msg);
    } else {
      AERROR << "Component object has been destroyed.";
    }
    auto end_time = Time::Now().ToMicrosecond();
    // sampling proc latency and cyber latency in microsecond
    uint64_t process_start_time;
    statistics::Statistics::Instance()->SamplingProcLatency<
                        uint64_t>(*role_attr, end_time-start_time);
    if (statistics::Statistics::Instance()->GetProcStatus(
          *role_attr, &process_start_time) && (
                        start_time-process_start_time) > 0) {
      statistics::Statistics::Instance()->SamplingCyberLatency(
                        *role_attr, start_time-process_start_time);
    }
  };

  std::shared_ptr<Reader<M0>> reader = nullptr;

  if (cyber_likely(is_reality_mode)) {
    reader = node_->CreateReader<M0>(reader_cfg);
  } else {
    reader = node_->CreateReader<M0>(reader_cfg, func);
  }

  if (reader == nullptr) {
    AERROR << "Component create reader failed.";
    return false;
  }
  readers_.emplace_back(std::move(reader));

  if (cyber_unlikely(!is_reality_mode)) {
    return true;
  }

  data::VisitorConfig conf = {readers_[0]->ChannelId(),
                              readers_[0]->PendingQueueSize()};
  auto dv = std::make_shared<data::DataVisitor<M0>>(conf);
  croutine::RoutineFactory factory =
      croutine::CreateRoutineFactory<M0>(func, dv);
  auto sched = scheduler::Instance();
  return sched->CreateTask(factory, node_->Name());
}

template <typename M0, typename M1>
bool Component<M0, M1, NullType, NullType>::Process(
    const std::shared_ptr<M0>& msg0, const std::shared_ptr<M1>& msg1) {
  if (is_shutdown_.load()) {
    return true;
  }
  return Proc(msg0, msg1);
}

template <typename M0, typename M1>
bool Component<M0, M1, NullType, NullType>::Initialize(
    const ComponentConfig& config) {
  node_.reset(new Node(config.name()));
  LoadConfigFiles(config);

  if (config.readers_size() < 2) {
    AERROR << "Invalid config file: too few readers.";
    return false;
  }

  if (!Init()) {
    AERROR << "Component Init() failed.";
    return false;
  }

  bool is_reality_mode = GlobalData::Instance()->IsRealityMode();

  ReaderConfig reader_cfg;
  reader_cfg.channel_name = config.readers(1).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(1).qos_profile());
  reader_cfg.pending_queue_size = config.readers(1).pending_queue_size();

  auto reader1 = node_->template CreateReader<M1>(reader_cfg);

  reader_cfg.channel_name = config.readers(0).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(0).qos_profile());
  reader_cfg.pending_queue_size = config.readers(0).pending_queue_size();

  auto role_attr = std::make_shared<proto::RoleAttributes>();
  role_attr->set_node_name(config.name());
  role_attr->set_channel_name(config.readers(0).channel());

  std::shared_ptr<Reader<M0>> reader0 = nullptr;
  if (cyber_likely(is_reality_mode)) {
    reader0 = node_->template CreateReader<M0>(reader_cfg);
  } else {
    std::weak_ptr<Component<M0, M1>> self =
        std::dynamic_pointer_cast<Component<M0, M1>>(shared_from_this());

    auto blocker1 = blocker::BlockerManager::Instance()->GetBlocker<M1>(
        config.readers(1).channel());

    auto func = [self, blocker1, role_attr](const std::shared_ptr<M0>& msg0) {
      auto start_time = Time::Now().ToMicrosecond();
      auto ptr = self.lock();
      if (ptr) {
        if (!blocker1->IsPublishedEmpty()) {
          auto msg1 = blocker1->GetLatestPublishedPtr();
          ptr->Process(msg0, msg1);
          auto end_time = Time::Now().ToMicrosecond();
          // sampling proc latency and cyber latency in microsecond
          uint64_t process_start_time;
          statistics::Statistics::Instance()->SamplingProcLatency<
                            uint64_t>(*role_attr, end_time-start_time);
          if (statistics::Statistics::Instance()->GetProcStatus(
                *role_attr, &process_start_time) && (
                                  start_time-process_start_time) > 0) {
            statistics::Statistics::Instance()->SamplingCyberLatency(
                              *role_attr, start_time-process_start_time);
          }
        }
      } else {
        AERROR << "Component object has been destroyed.";
      }
    };

    reader0 = node_->template CreateReader<M0>(reader_cfg, func);
  }
  if (reader0 == nullptr || reader1 == nullptr) {
    AERROR << "Component create reader failed.";
    return false;
  }
  readers_.push_back(std::move(reader0));
  readers_.push_back(std::move(reader1));

  if (cyber_unlikely(!is_reality_mode)) {
    return true;
  }

  auto sched = scheduler::Instance();
  std::weak_ptr<Component<M0, M1>> self =
      std::dynamic_pointer_cast<Component<M0, M1>>(shared_from_this());
  auto func = [self, role_attr](const std::shared_ptr<M0>& msg0,
                     const std::shared_ptr<M1>& msg1) {
    auto start_time = Time::Now().ToMicrosecond();
    auto ptr = self.lock();
    if (ptr) {
      ptr->Process(msg0, msg1);
      auto end_time = Time::Now().ToMicrosecond();
      // sampling proc latency and cyber latency in microsecond
      uint64_t process_start_time;
      statistics::Statistics::Instance()->SamplingProcLatency<
                        uint64_t>(*role_attr, end_time-start_time);
      if (statistics::Statistics::Instance()->GetProcStatus(
            *role_attr, &process_start_time) && (
                                start_time-process_start_time) > 0) {
        statistics::Statistics::Instance()->SamplingCyberLatency(
                          *role_attr, start_time-process_start_time);
      }
    } else {
      AERROR << "Component object has been destroyed.";
    }
  };

  std::vector<data::VisitorConfig> config_list;
  for (auto& reader : readers_) {
    config_list.emplace_back(reader->ChannelId(), reader->PendingQueueSize());
  }
  auto dv = std::make_shared<data::DataVisitor<M0, M1>>(config_list);
  croutine::RoutineFactory factory =
      croutine::CreateRoutineFactory<M0, M1>(func, dv);
  return sched->CreateTask(factory, node_->Name());
}

template <typename M0, typename M1, typename M2>
bool Component<M0, M1, M2, NullType>::Process(const std::shared_ptr<M0>& msg0,
                                              const std::shared_ptr<M1>& msg1,
                                              const std::shared_ptr<M2>& msg2) {
  if (is_shutdown_.load()) {
    return true;
  }
  return Proc(msg0, msg1, msg2);
}

template <typename M0, typename M1, typename M2>
bool Component<M0, M1, M2, NullType>::Initialize(
    const ComponentConfig& config) {
  node_.reset(new Node(config.name()));
  LoadConfigFiles(config);

  if (config.readers_size() < 3) {
    AERROR << "Invalid config file: too few readers.";
    return false;
  }

  if (!Init()) {
    AERROR << "Component Init() failed.";
    return false;
  }

  bool is_reality_mode = GlobalData::Instance()->IsRealityMode();

  ReaderConfig reader_cfg;
  reader_cfg.channel_name = config.readers(1).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(1).qos_profile());
  reader_cfg.pending_queue_size = config.readers(1).pending_queue_size();

  auto reader1 = node_->template CreateReader<M1>(reader_cfg);

  reader_cfg.channel_name = config.readers(2).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(2).qos_profile());
  reader_cfg.pending_queue_size = config.readers(2).pending_queue_size();

  auto reader2 = node_->template CreateReader<M2>(reader_cfg);

  reader_cfg.channel_name = config.readers(0).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(0).qos_profile());
  reader_cfg.pending_queue_size = config.readers(0).pending_queue_size();

  auto role_attr = std::make_shared<proto::RoleAttributes>();
  role_attr->set_node_name(config.name());
  role_attr->set_channel_name(config.readers(0).channel());

  std::shared_ptr<Reader<M0>> reader0 = nullptr;
  if (cyber_likely(is_reality_mode)) {
    reader0 = node_->template CreateReader<M0>(reader_cfg);
  } else {
    std::weak_ptr<Component<M0, M1, M2, NullType>> self =
        std::dynamic_pointer_cast<Component<M0, M1, M2, NullType>>(
            shared_from_this());

    auto blocker1 = blocker::BlockerManager::Instance()->GetBlocker<M1>(
        config.readers(1).channel());
    auto blocker2 = blocker::BlockerManager::Instance()->GetBlocker<M2>(
        config.readers(2).channel());

    auto func = [self, blocker1, blocker2, role_attr](
                        const std::shared_ptr<M0>& msg0) {
      auto start_time = Time::Now().ToMicrosecond();
      auto ptr = self.lock();
      if (ptr) {
        if (!blocker1->IsPublishedEmpty() && !blocker2->IsPublishedEmpty()) {
          auto msg1 = blocker1->GetLatestPublishedPtr();
          auto msg2 = blocker2->GetLatestPublishedPtr();
          ptr->Process(msg0, msg1, msg2);
          auto end_time = Time::Now().ToMicrosecond();
          // sampling proc latency and cyber latency in microsecond
          uint64_t process_start_time;
          statistics::Statistics::Instance()->SamplingProcLatency<
                              uint64_t>(*role_attr, end_time-start_time);
          if (statistics::Statistics::Instance()->GetProcStatus(
                *role_attr, &process_start_time) && (
                                    start_time-process_start_time) > 0) {
            statistics::Statistics::Instance()->SamplingCyberLatency(
                              *role_attr, start_time-process_start_time);
          }
        }
      } else {
        AERROR << "Component object has been destroyed.";
      }
    };

    reader0 = node_->template CreateReader<M0>(reader_cfg, func);
  }

  if (reader0 == nullptr || reader1 == nullptr || reader2 == nullptr) {
    AERROR << "Component create reader failed.";
    return false;
  }
  readers_.push_back(std::move(reader0));
  readers_.push_back(std::move(reader1));
  readers_.push_back(std::move(reader2));

  if (cyber_unlikely(!is_reality_mode)) {
    return true;
  }

  auto sched = scheduler::Instance();
  std::weak_ptr<Component<M0, M1, M2, NullType>> self =
      std::dynamic_pointer_cast<Component<M0, M1, M2, NullType>>(
          shared_from_this());
  auto func = [self, role_attr](const std::shared_ptr<M0>& msg0,
                                const std::shared_ptr<M1>& msg1,
                                const std::shared_ptr<M2>& msg2) {
    auto start_time = Time::Now().ToMicrosecond();
    auto ptr = self.lock();
    if (ptr) {
      ptr->Process(msg0, msg1, msg2);
      auto end_time = Time::Now().ToMicrosecond();
      // sampling proc latency and cyber latency in microsecond
      uint64_t process_start_time;
      statistics::Statistics::Instance()->SamplingProcLatency<
                        uint64_t>(*role_attr, end_time-start_time);
      if (statistics::Statistics::Instance()->GetProcStatus(
            *role_attr, &process_start_time) && (
                                  start_time-process_start_time) > 0) {
        statistics::Statistics::Instance()->SamplingCyberLatency(
                            *role_attr, start_time-process_start_time);
      }
    } else {
      AERROR << "Component object has been destroyed.";
    }
  };

  std::vector<data::VisitorConfig> config_list;
  for (auto& reader : readers_) {
    config_list.emplace_back(reader->ChannelId(), reader->PendingQueueSize());
  }
  auto dv = std::make_shared<data::DataVisitor<M0, M1, M2>>(config_list);
  croutine::RoutineFactory factory =
      croutine::CreateRoutineFactory<M0, M1, M2>(func, dv);
  return sched->CreateTask(factory, node_->Name());
}

template <typename M0, typename M1, typename M2, typename M3>
bool Component<M0, M1, M2, M3>::Process(const std::shared_ptr<M0>& msg0,
                                        const std::shared_ptr<M1>& msg1,
                                        const std::shared_ptr<M2>& msg2,
                                        const std::shared_ptr<M3>& msg3) {
  if (is_shutdown_.load()) {
    return true;
  }
  // DEBUG: astro-svgfigure pipeline stage marker — 4-channel cell Proc() entry.
  // This is the fusion point where skeleton_signal (M0), force_field (M1),
  // palette (M2), and z_layers (M3) are merged into a single constraint dispatch.
  // astro_channel_ready_mask is rebuilt each call: bit i is set iff the
  // corresponding shared_ptr is non-null, allowing partial-delivery diagnostics
  // without blocking the dispatch (the DataVisitor guarantees all four are
  // present in reality mode, but simulation mode may deliver stale blockers).
  ++astro_component_epoch;
  astro_channel_ready_mask = ((msg0 ? 0x1 : 0) |
                              (msg1 ? 0x2 : 0) |
                              (msg2 ? 0x4 : 0) |
                              (msg3 ? 0x8 : 0));
  fprintf(stderr,
          "[ASTRO-COMPONENT] proc-enter | epoch=%d ch_mask=0x%x "
          "skeleton=%s force_field=%s palette=%s z_layers=%s\n",
          astro_component_epoch,
          astro_channel_ready_mask,
          msg0 ? "ok" : "null",   // skeleton_signal channel
          msg1 ? "ok" : "null",   // force_field channel
          msg2 ? "ok" : "null",   // palette channel
          msg3 ? "ok" : "null");  // z_layers channel
  return Proc(msg0, msg1, msg2, msg3);
}

template <typename M0, typename M1, typename M2, typename M3>
bool Component<M0, M1, M2, M3>::Initialize(const ComponentConfig& config) {
  node_.reset(new Node(config.name()));
  LoadConfigFiles(config);

  // DEBUG: astro-svgfigure pipeline stage marker — 4-channel Component::Initialize().
  // Maps to the astro cell boot sequence: four pubsub readers are bound here,
  // one per cell signal channel.  Channel-to-semantic mapping:
  //   readers(0) → skeleton_signal  : bone/joint positions from animation rig
  //   readers(1) → force_field      : repulsion/attraction vectors for bbox solver
  //   readers(2) → palette          : color-space slot assignments for cell render
  //   readers(3) → z_layers         : stacking order commands for FAstroZLayerRegistry
  // The DataVisitor<M0,M1,M2,M3> created at the end synchronises all four so
  // that Proc() fires only when every channel has a fresh message — analogous
  // to the ShadowRendering bbox collision check that requires all z-layer
  // participants to have reported their positions before resolving overlaps.
  fprintf(stderr,
          "[ASTRO-COMPONENT] init-start | node=%s readers=%d "
          "ch0(skeleton)=%s ch1(force_field)=%s ch2(palette)=%s ch3(z_layers)=%s\n",
          config.name().c_str(),
          config.readers_size(),
          config.readers_size() > 0 ? config.readers(0).channel().c_str() : "?",
          config.readers_size() > 1 ? config.readers(1).channel().c_str() : "?",
          config.readers_size() > 2 ? config.readers(2).channel().c_str() : "?",
          config.readers_size() > 3 ? config.readers(3).channel().c_str() : "?");

  if (config.readers_size() < 4) {
    AERROR << "Invalid config file: too few readers_." << std::endl;
    return false;
  }

  if (!Init()) {
    AERROR << "Component Init() failed." << std::endl;
    return false;
  }

  bool is_reality_mode = GlobalData::Instance()->IsRealityMode();

  ReaderConfig reader_cfg;
  reader_cfg.channel_name = config.readers(1).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(1).qos_profile());
  reader_cfg.pending_queue_size = config.readers(1).pending_queue_size();

  auto reader1 = node_->template CreateReader<M1>(reader_cfg);

  reader_cfg.channel_name = config.readers(2).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(2).qos_profile());
  reader_cfg.pending_queue_size = config.readers(2).pending_queue_size();

  auto reader2 = node_->template CreateReader<M2>(reader_cfg);

  reader_cfg.channel_name = config.readers(3).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(3).qos_profile());
  reader_cfg.pending_queue_size = config.readers(3).pending_queue_size();

  auto reader3 = node_->template CreateReader<M3>(reader_cfg);

  reader_cfg.channel_name = config.readers(0).channel();
  reader_cfg.qos_profile.CopyFrom(config.readers(0).qos_profile());
  reader_cfg.pending_queue_size = config.readers(0).pending_queue_size();

  auto role_attr = std::make_shared<proto::RoleAttributes>();
  role_attr->set_node_name(config.name());
  role_attr->set_channel_name(config.readers(0).channel());

  std::shared_ptr<Reader<M0>> reader0 = nullptr;
  if (cyber_likely(is_reality_mode)) {
    reader0 = node_->template CreateReader<M0>(reader_cfg);
  } else {
    std::weak_ptr<Component<M0, M1, M2, M3>> self =
        std::dynamic_pointer_cast<Component<M0, M1, M2, M3>>(
            shared_from_this());

    auto blocker1 = blocker::BlockerManager::Instance()->GetBlocker<M1>(
        config.readers(1).channel());
    auto blocker2 = blocker::BlockerManager::Instance()->GetBlocker<M2>(
        config.readers(2).channel());
    auto blocker3 = blocker::BlockerManager::Instance()->GetBlocker<M3>(
        config.readers(3).channel());

    auto func = [self, blocker1, blocker2,
                 blocker3, role_attr](const std::shared_ptr<M0>& msg0) {
      auto start_time = Time::Now().ToMicrosecond();
      auto ptr = self.lock();
      if (ptr) {
        if (!blocker1->IsPublishedEmpty() && !blocker2->IsPublishedEmpty() &&
            !blocker3->IsPublishedEmpty()) {
          auto msg1 = blocker1->GetLatestPublishedPtr();
          auto msg2 = blocker2->GetLatestPublishedPtr();
          auto msg3 = blocker3->GetLatestPublishedPtr();
          // DEBUG: astro-svgfigure pipeline stage marker — simulation-mode
          // 4-channel fusion via blockers.  All three secondary channels
          // (force_field/palette/z_layers) have published; skeleton_signal
          // (msg0) arrived on the trigger reader.  This path mirrors the
          // astro cell constraint dispatch in non-reality mode: blockers
          // stand in for the DataVisitor, holding the last published msg
          // from each channel so Proc() can fuse the quad-tuple offline.
          ++astro_proc_fusion_count;
          fprintf(stderr,
                  "[ASTRO-COMPONENT] sim-fusion | node=%s fusion_count=%d "
                  "epoch=%d skel=ok force=ok palette=ok zlayers=ok\n",
                  role_attr->node_name().c_str(),
                  astro_proc_fusion_count,
                  astro_component_epoch);
          ptr->Process(msg0, msg1, msg2, msg3);
          auto end_time = Time::Now().ToMicrosecond();
          // sampling proc latency and cyber latency in microsecond
          uint64_t process_start_time;
          statistics::Statistics::Instance()->SamplingProcLatency<
                            uint64_t>(*role_attr, end_time-start_time);
          if (statistics::Statistics::Instance()->GetProcStatus(
                *role_attr, &process_start_time) && (
                                      start_time-process_start_time) > 0) {
            statistics::Statistics::Instance()->SamplingCyberLatency(
                                *role_attr, start_time-process_start_time);
          }
        }
      } else {
        AERROR << "Component object has been destroyed.";
      }
    };

    reader0 = node_->template CreateReader<M0>(reader_cfg, func);
  }

  if (reader0 == nullptr || reader1 == nullptr || reader2 == nullptr ||
      reader3 == nullptr) {
    AERROR << "Component create reader failed." << std::endl;
    return false;
  }
  readers_.push_back(std::move(reader0));
  readers_.push_back(std::move(reader1));
  readers_.push_back(std::move(reader2));
  readers_.push_back(std::move(reader3));

  if (cyber_unlikely(!is_reality_mode)) {
    return true;
  }

  auto sched = scheduler::Instance();
  std::weak_ptr<Component<M0, M1, M2, M3>> self =
      std::dynamic_pointer_cast<Component<M0, M1, M2, M3>>(shared_from_this());
  auto func =
      [self, role_attr](const std::shared_ptr<M0>& msg0,
                        const std::shared_ptr<M1>& msg1,
                        const std::shared_ptr<M2>& msg2,
                        const std::shared_ptr<M3>& msg3) {
        auto start_time = Time::Now().ToMicrosecond();
        auto ptr = self.lock();
        if (ptr) {
          // DEBUG: astro-svgfigure pipeline stage marker — reality-mode
          // DataVisitor 4-channel fusion coroutine fire.  DataVisitor has
          // already synchronised all four channels; this lambda is the
          // croutine scheduled by sched->CreateTask().  In astro terms,
          // this is the constraint-solver tick: skeleton_signal positions
          // are applied first (M0), then force_field repulsion is integrated
          // (M1), palette slots committed (M2), and z_layer registry updated
          // (M3).  astro_zlayer_commit_count tracks cumulative z-layer
          // dispatches so the registry can detect epoch wrap-around.
          ++astro_proc_fusion_count;
          ++astro_zlayer_commit_count;
          fprintf(stderr,
                  "[ASTRO-COMPONENT] reality-fusion | node=%s fusion=%d "
                  "epoch=%d zlayer_commits=%d ch_mask=0x%x\n",
                  role_attr->node_name().c_str(),
                  astro_proc_fusion_count,
                  astro_component_epoch,
                  astro_zlayer_commit_count,
                  0xf /* all 4 channels guaranteed by DataVisitor */);
          ptr->Process(msg0, msg1, msg2, msg3);
          auto end_time = Time::Now().ToMicrosecond();
          // sampling proc latency and cyber latency in microsecond
          uint64_t process_start_time;
          statistics::Statistics::Instance()->SamplingProcLatency<
                            uint64_t>(*role_attr, end_time-start_time);
          if (statistics::Statistics::Instance()->GetProcStatus(
                *role_attr, &process_start_time) && (
                                      start_time-process_start_time) > 0) {
            statistics::Statistics::Instance()->SamplingCyberLatency(
                              *role_attr, start_time-process_start_time);
          }
        } else {
          AERROR << "Component object has been destroyed." << std::endl;
        }
      };

  std::vector<data::VisitorConfig> config_list;
  for (auto& reader : readers_) {
    config_list.emplace_back(reader->ChannelId(), reader->PendingQueueSize());
  }
  auto dv = std::make_shared<data::DataVisitor<M0, M1, M2, M3>>(config_list);
  croutine::RoutineFactory factory =
      croutine::CreateRoutineFactory<M0, M1, M2, M3>(func, dv);
  // DEBUG: astro-svgfigure pipeline stage marker — CreateTask fires the
  // DataVisitor coroutine into the cyber scheduler.  From this point the
  // 4-channel cell fusion loop is live: every time skeleton_signal (M0),
  // force_field (M1), palette (M2), and z_layers (M3) all carry a fresh
  // message, the scheduler wakes the coroutine and Proc() runs one full
  // constraint-solve tick.  Log the init completion so we can correlate
  // the first [ASTRO-COMPONENT] proc-enter line with this bootstrap.
  fprintf(stderr,
          "[ASTRO-COMPONENT] init-done | node=%s task=%s "
          "skeleton_ch=%s force_ch=%s palette_ch=%s zlayers_ch=%s\n",
          config.name().c_str(),
          node_->Name().c_str(),
          config.readers(0).channel().c_str(),
          config.readers(1).channel().c_str(),
          config.readers(2).channel().c_str(),
          config.readers(3).channel().c_str());
  return sched->CreateTask(factory, node_->Name());
}

#define CYBER_REGISTER_COMPONENT(name) \
  CLASS_LOADER_REGISTER_CLASS(name, apollo::cyber::ComponentBase)

}  // namespace cyber
}  // namespace apollo

#endif  // CYBER_COMPONENT_COMPONENT_H_
