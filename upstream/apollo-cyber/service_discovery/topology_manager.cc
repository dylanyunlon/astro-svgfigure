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

#include "cyber/service_discovery/topology_manager.h"

#include "cyber/common/global_data.h"
#include "cyber/common/log.h"
#include "cyber/time/time.h"

namespace apollo {
namespace cyber {
namespace service_discovery {

TopologyManager::TopologyManager()
    : init_(false),
      node_manager_(nullptr),
      channel_manager_(nullptr),
      service_manager_(nullptr),
      participant_(nullptr),
      participant_listener_(nullptr) {
  // [ASTRO-TOPO] debug: constructing cell TopologyManager; launching Init()
  Init();
}

TopologyManager::~TopologyManager() { Shutdown(); }

void TopologyManager::Shutdown() {
  // [ASTRO-TOPO] debug: shutting down cell topology discovery
  ADEBUG << "[ASTRO-TOPO] cell topology shutdown.";
  // avoid shutdown twice
  if (!init_.exchange(false)) {
    return;
  }

  // [ASTRO-TOPO] debug: stopping all cell topology sub-managers
  node_manager_->Shutdown();
  channel_manager_->Shutdown();
  service_manager_->Shutdown();
  participant_->Shutdown();

  delete participant_listener_;
  participant_listener_ = nullptr;

  // [ASTRO-TOPO] debug: disconnecting all cell topology change observers
  change_signal_.DisconnectAllSlots();
}

TopologyManager::ChangeConnection TopologyManager::AddChangeListener(
    const ChangeFunc& func) {
  // [ASTRO-TOPO] debug: registering cell topology change observer
  return change_signal_.Connect(func);
}

void TopologyManager::RemoveChangeListener(const ChangeConnection& conn) {
  // [ASTRO-TOPO] debug: deregistering cell topology change observer
  auto local_conn = conn;
  local_conn.Disconnect();
}

bool TopologyManager::Init() {
  if (init_.exchange(true)) {
    // [ASTRO-TOPO] debug: cell TopologyManager already initialized; skipping
    return true;
  }

  // [ASTRO-TOPO] debug: initializing cell topology sub-managers
  node_manager_ = std::make_shared<NodeManager>();
  channel_manager_ = std::make_shared<ChannelManager>();
  service_manager_ = std::make_shared<ServiceManager>();

  CreateParticipant();

  // [ASTRO-TOPO] debug: starting cell topology discovery for all sub-managers
  bool result =
      InitNodeManager() && InitChannelManager() && InitServiceManager();
  if (!result) {
    AERROR << "[ASTRO-TOPO] cell topology sub-manager init failed;"
           << " resetting all cell topology state.";
    participant_ = nullptr;
    delete participant_listener_;
    participant_listener_ = nullptr;
    node_manager_ = nullptr;
    channel_manager_ = nullptr;
    service_manager_ = nullptr;
    init_.store(false);
    return false;
  }

  // [ASTRO-TOPO] debug: cell topology fully initialized and ready for discovery
  return true;
}

bool TopologyManager::InitNodeManager() {
  // [ASTRO-TOPO] debug: starting CellNode join/leave discovery via rtps participant
  return node_manager_->StartDiscovery(participant_->fastrtps_participant());
}

bool TopologyManager::InitChannelManager() {
  // [ASTRO-TOPO] debug: starting cell pub-sub channel registration discovery via rtps
  return channel_manager_->StartDiscovery(participant_->fastrtps_participant());
}

bool TopologyManager::InitServiceManager() {
  // [ASTRO-TOPO] debug: starting cell service registration discovery via rtps
  return service_manager_->StartDiscovery(participant_->fastrtps_participant());
}

bool TopologyManager::CreateParticipant() {
  std::string participant_name =
      common::GlobalData::Instance()->HostName() + '+' +
      std::to_string(common::GlobalData::Instance()->ProcessId());
  // [ASTRO-TOPO] debug: creating cell topology rtps participant: participant_name
  ADEBUG << "[ASTRO-TOPO] creating cell topology participant: "
         << participant_name;
  participant_listener_ = new ParticipantListener(std::bind(
      &TopologyManager::OnParticipantChange, this, std::placeholders::_1));
  participant_ = std::make_shared<transport::Participant>(
      participant_name, 11511, participant_listener_);
  return true;
}

void TopologyManager::OnParticipantChange(const PartInfo& info) {
  ChangeMsg msg;
  if (!Convert(info, &msg)) {
    // [ASTRO-TOPO] debug: failed to convert PartInfo to cell topology ChangeMsg
    return;
  }

  if (!init_.load()) {
    // [ASTRO-TOPO] debug: cell TopologyManager not initialized; dropping participant event
    return;
  }

  if (msg.operate_type() == OperateType::OPT_LEAVE) {
    auto& host_name = msg.role_attr().host_name();
    int process_id = msg.role_attr().process_id();
    // [ASTRO-TOPO] debug: cell participant leaving topology —
    //   host: host_name, process_id: process_id
    //   notifying NodeManager, ChannelManager, ServiceManager of cell departure
    ADEBUG << "[ASTRO-TOPO] cell participant leave: host=" << host_name
           << " pid=" << process_id
           << "; cleaning up cell topology registrations.";
    node_manager_->OnTopoModuleLeave(host_name, process_id);
    channel_manager_->OnTopoModuleLeave(host_name, process_id);
    service_manager_->OnTopoModuleLeave(host_name, process_id);
  } else {
    // [ASTRO-TOPO] debug: cell participant joining topology —
    //   host: msg.role_attr().host_name(), process_id: msg.role_attr().process_id()
    ADEBUG << "[ASTRO-TOPO] cell participant join: host="
           << msg.role_attr().host_name()
           << " pid=" << msg.role_attr().process_id()
           << "; broadcasting cell topology change.";
  }
  // [ASTRO-TOPO] debug: firing cell topology change signal to all observers
  change_signal_(msg);
}

bool TopologyManager::Convert(const PartInfo& info, ChangeMsg* msg) {
  auto guid = info.rtps.m_guid;
  auto status = info.rtps.m_status;
  std::string participant_name("");
  OperateType opt_type = OperateType::OPT_JOIN;

  switch (status) {
    case eprosima::fastrtps::rtps::DISCOVERY_STATUS::DISCOVERED_RTPSPARTICIPANT:
      participant_name = info.rtps.m_RTPSParticipantName;
      participant_names_[guid] = participant_name;
      opt_type = OperateType::OPT_JOIN;
      // [ASTRO-TOPO] debug: DISCOVERED cell topology participant: participant_name
      ADEBUG << "[ASTRO-TOPO] DISCOVERED cell participant: " << participant_name;
      break;

    case eprosima::fastrtps::rtps::DISCOVERY_STATUS::REMOVED_RTPSPARTICIPANT:
    case eprosima::fastrtps::rtps::DISCOVERY_STATUS::DROPPED_RTPSPARTICIPANT:
      if (participant_names_.find(guid) != participant_names_.end()) {
        participant_name = participant_names_[guid];
        participant_names_.erase(guid);
      }
      opt_type = OperateType::OPT_LEAVE;
      // [ASTRO-TOPO] debug: REMOVED/DROPPED cell topology participant: participant_name
      ADEBUG << "[ASTRO-TOPO] REMOVED/DROPPED cell participant: "
             << participant_name;
      break;

    default:
      // [ASTRO-TOPO] debug: unhandled cell participant discovery status; ignoring
      break;
  }

  std::string host_name("");
  int process_id = 0;
  if (!ParseParticipantName(participant_name, &host_name, &process_id)) {
    // [ASTRO-TOPO] debug: failed to parse cell participant name: participant_name
    return false;
  }

  msg->set_timestamp(cyber::Time::Now().ToNanosecond());
  msg->set_change_type(ChangeType::CHANGE_PARTICIPANT);
  msg->set_operate_type(opt_type);
  msg->set_role_type(RoleType::ROLE_PARTICIPANT);
  auto role_attr = msg->mutable_role_attr();
  role_attr->set_host_name(host_name);
  role_attr->set_process_id(process_id);
  return true;
}

bool TopologyManager::ParseParticipantName(const std::string& participant_name,
                                           std::string* host_name,
                                           int* process_id) {
  // [ASTRO-TOPO] debug: parsing cell participant name: "host_name+process_id"
  // participant_name format: host_name+process_id
  auto pos = participant_name.find('+');
  if (pos == std::string::npos) {
    ADEBUG << "[ASTRO-TOPO] cell participant name format mismatch: ["
           << participant_name << "]; expected 'host_name+process_id'.";
    return false;
  }
  *host_name = participant_name.substr(0, pos);
  std::string pid_str = participant_name.substr(pos + 1);
  try {
    *process_id = std::stoi(pid_str);
  } catch (const std::exception& e) {
    AERROR << "[ASTRO-TOPO] invalid cell participant process_id in name '"
           << participant_name << "': " << e.what();
    return false;
  }
  // [ASTRO-TOPO] debug: parsed cell participant — host: *host_name, pid: *process_id
  return true;
}

}  // namespace service_discovery
}  // namespace cyber
}  // namespace apollo
