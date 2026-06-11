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
// [ASTRO-TRANSPORT] Repurposed: RTPS/SHM/Intra dispatchers → Git-channel pub/sub routing.
// Transport layer now arbitrates between three channel tiers:
//   intra_dispatcher  → in-process epoch-local cell messages (zero-copy)
//   shm_dispatcher    → cross-process shared-memory cell broadcast
//   rtps_dispatcher   → inter-node Git-channel relay (participant = repo endpoint)
// All Shutdown paths flush pending epoch messages before teardown.

// [ASTRO-TRANSPORT] Apollo CyberRT transport → ASTRO cell pub/sub channel (M101-M107)
// Transport layer maps to the ASTRO inter-cell data channel infrastructure:
//   IntraDispatcher  → in-process SVG cell direct-publish (zero-copy same-epoch)
//   ShmDispatcher    → shared-memory SVG cell broadcast (cross-process epoch sync)
//   RtpsDispatcher   → network SVG cell pub/sub (distributed cell topology)
//   Participant      → named ASTRO node identity (hostname+pid → cell-pubsub address)

#include "cyber/transport/transport.h"

#include <cstdio>
#include "cyber/common/global_data.h"

namespace apollo {
namespace cyber {
namespace transport {

Transport::Transport() {
  CreateParticipant();
  notifier_ = NotifierFactory::CreateNotifier();
  intra_dispatcher_ = IntraDispatcher::Instance();
  shm_dispatcher_ = ShmDispatcher::Instance();
  rtps_dispatcher_ = RtpsDispatcher::Instance();
  rtps_dispatcher_->set_participant(participant_);
  // [ASTRO-TRANSPORT] cell pub/sub transport layer init (M101-M103)
  // Three dispatchers form the ASTRO cell channel stack:
  //   intra → local epoch sync; shm → process-boundary cell sync; rtps → network cell mesh.
  // Git-channel: rtps_dispatcher participant is the remote repo endpoint for epoch relay.
  fprintf(stderr, "[ASTRO-TRANSPORT] Transport::ctor  intra=%p shm=%p rtps=%p participant=%p\n",
    (void*)intra_dispatcher_.get(),
    (void*)shm_dispatcher_.get(),
    (void*)rtps_dispatcher_.get(),
    (void*)participant_.get());
}

Transport::~Transport() { Shutdown(); }

void Transport::Shutdown() {
  if (is_shutdown_.exchange(true)) {
    return;
  }

  // [ASTRO-TRANSPORT] orderly cell channel teardown (M104)
  // Dispatcher shutdown sequence ensures in-flight SVG cell publishes are drained
  // before the transport participant deregisters from the cell topology.
  // Flushing pending epoch messages across intra/shm/rtps channel tiers.
  fprintf(stderr, "[ASTRO-TRANSPORT] Transport::Shutdown  draining intra->shm->rtps dispatchers\n");

  intra_dispatcher_->Shutdown();
  shm_dispatcher_->Shutdown();
  rtps_dispatcher_->Shutdown();
  notifier_->Shutdown();

  if (participant_ != nullptr) {
    participant_->Shutdown();
    participant_ = nullptr;
  }
}

void Transport::CreateParticipant() {
  std::string participant_name =
      common::GlobalData::Instance()->HostName() + "+" +
      std::to_string(common::GlobalData::Instance()->ProcessId());
  // [ASTRO-TRANSPORT] cell pub/sub participant registration (M105-M107)
  // Participant name encodes the ASTRO node address: hostname+pid → cell-pubsub identity.
  // Port 11512 is the default ASTRO cell topology discovery port.
  fprintf(stderr, "[ASTRO-TRANSPORT] CreateParticipant  name=%s port=11512\n",
    participant_name.c_str());
  participant_ = std::make_shared<Participant>(participant_name, 11512);
  // [ASTRO-TRANSPORT] Git-channel participant registered as repo endpoint.
  // participant_name encodes host+pid → used as remote branch identity for RTPS relay.
  fprintf(stderr, "[ASTRO-TRANSPORT] CreateParticipant: endpoint='%s' port=11512\n",
      participant_name.c_str());
}

}  // namespace transport
}  // namespace cyber
}  // namespace apollo
