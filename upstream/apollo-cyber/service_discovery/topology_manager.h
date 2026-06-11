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

#ifndef CYBER_SERVICE_DISCOVERY_TOPOLOGY_MANAGER_H_
#define CYBER_SERVICE_DISCOVERY_TOPOLOGY_MANAGER_H_

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "cyber/base/signal.h"
#include "cyber/common/macros.h"
#include "cyber/service_discovery/communication/participant_listener.h"
#include "cyber/service_discovery/specific_manager/channel_manager.h"
#include "cyber/service_discovery/specific_manager/node_manager.h"
#include "cyber/service_discovery/specific_manager/service_manager.h"
#include "cyber/transport/rtps/participant.h"

namespace apollo {
namespace cyber {
namespace service_discovery {

class NodeManager;
using NodeManagerPtr = std::shared_ptr<NodeManager>;

class ChannelManager;
using ChannelManagerPtr = std::shared_ptr<ChannelManager>;

class ServiceManager;
using ServiceManagerPtr = std::shared_ptr<ServiceManager>;

/**
 * @class TopologyManager
 * @brief Cell topology discovery and registration manager for Astro/Cyber RT.
 *
 * TopologyManager maintains the global cell topology: each CellNode, its
 * pub/sub channels, and its services are registered and discovered here.
 * The topology is represented as a directed graph:
 *   - CellNode: container/vertex in the cell topology graph.
 *   - Channel: directed edge from Writer (upstream cell) to Reader (downstream cell).
 *   - Service: directed edge from Server cell to Client cell.
 *
 * Three sub-managers handle cell topology registration:
 *   - NodeManager:    tracks which CellNodes are active in the topology.
 *   - ChannelManager: tracks cell pub-sub channel registrations (Writers/Readers).
 *   - ServiceManager: tracks cell service registrations (Servers/Clients).
 *
 * TopologyManager uses fast-rtps Participant for distributed cell topology
 * broadcast: CellNodes broadcast Join/Leave events to all topology participants.
 * You can register a `ChangeFunc` to observe cell topology changes.
 *
 * [ASTRO-TOPO] Cell topology registration flow:
 *   1. TopologyManager::Init() creates NodeManager, ChannelManager, ServiceManager.
 *   2. CreateParticipant() sets up the rtps participant for cell topology broadcast.
 *   3. InitNodeManager/InitChannelManager/InitServiceManager start cell discovery.
 *   4. OnParticipantChange() fires when a cell participant joins or leaves the topology.
 *   5. change_signal_ notifies all registered ChangeFunc observers of cell topo changes.
 */
class TopologyManager {
 public:
  using ChangeSignal = base::Signal<const ChangeMsg&>;
  using ChangeFunc = std::function<void(const ChangeMsg&)>;
  using ChangeConnection = base::Connection<const ChangeMsg&>;
  using PartNameContainer =
      std::map<eprosima::fastrtps::rtps::GUID_t, std::string>;
  using PartInfo = eprosima::fastrtps::ParticipantDiscoveryInfo;

  virtual ~TopologyManager();

  /**
   * @brief Shutdown the TopologyManager and all cell topology sub-managers.
   *
   * [ASTRO-TOPO] debug: Shutdown stops cell topology discovery for all
   * registered NodeManager, ChannelManager, and ServiceManager instances.
   * Disconnects all change_signal_ observers.
   */
  void Shutdown();

  /**
   * @brief Register a `ChangeFunc` to observe cell topology changes.
   *
   * @param func observer callback invoked on cell topology Join/Leave events.
   * @return ChangeConnection connection handle; use to deregister the observer.
   *
   * [ASTRO-TOPO] debug: observer is connected to change_signal_. Multiple
   * observers can track cell topology changes simultaneously.
   */
  ChangeConnection AddChangeListener(const ChangeFunc& func);

  /**
   * @brief Deregister a cell topology change observer by connection handle.
   *
   * [ASTRO-TOPO] debug: disconnects the observer from change_signal_.
   */
  void RemoveChangeListener(const ChangeConnection& conn);

  /**
   * @brief Get shared_ptr for the cell NodeManager.
   *
   * [ASTRO-TOPO] debug: NodeManager tracks active CellNodes in the topology.
   */
  NodeManagerPtr& node_manager() { return node_manager_; }

  /**
   * @brief Get shared_ptr for the cell ChannelManager.
   *
   * [ASTRO-TOPO] debug: ChannelManager tracks cell pub-sub channel registrations.
   */
  ChannelManagerPtr& channel_manager() { return channel_manager_; }

  /**
   * @brief Get shared_ptr for the cell ServiceManager.
   *
   * [ASTRO-TOPO] debug: ServiceManager tracks cell service registrations.
   */
  ServiceManagerPtr& service_manager() { return service_manager_; }

 private:
  /// Initialize all cell topology sub-managers and the rtps participant.
  /// [ASTRO-TOPO] debug: Init() is idempotent; guarded by init_ atomic flag.
  bool Init();

  /// Initialize the cell NodeManager and start node discovery.
  /// [ASTRO-TOPO] debug: starts CellNode join/leave discovery via rtps.
  bool InitNodeManager();

  /// Initialize the cell ChannelManager and start channel discovery.
  /// [ASTRO-TOPO] debug: starts pub-sub channel registration discovery via rtps.
  bool InitChannelManager();

  /// Initialize the cell ServiceManager and start service discovery.
  /// [ASTRO-TOPO] debug: starts service registration discovery via rtps.
  bool InitServiceManager();

  /// Create the rtps participant for cell topology broadcast.
  /// [ASTRO-TOPO] debug: participant_name = host_name + '+' + process_id.
  bool CreateParticipant();

  /// Handle rtps participant discovery events (cell topology Join/Leave).
  /// [ASTRO-TOPO] debug: fires on any remote CellNode participant change.
  void OnParticipantChange(const PartInfo& info);

  /// Convert rtps PartInfo to a cell topology ChangeMsg.
  /// [ASTRO-TOPO] debug: maps DISCOVERED/REMOVED/DROPPED status to OPT_JOIN/OPT_LEAVE.
  bool Convert(const PartInfo& info, ChangeMsg* change_msg);

  /// Parse participant_name (host_name+process_id) from cell topology broadcast.
  /// [ASTRO-TOPO] debug: format is "host_name+process_id"; '+' is the delimiter.
  bool ParseParticipantName(const std::string& participant_name,
                            std::string* host_name, int* process_id);

  std::atomic<bool> init_;             ///< Is cell TopologyManager initialized
  NodeManagerPtr node_manager_;        ///< shared ptr of cell NodeManager
  ChannelManagerPtr channel_manager_;  ///< shared ptr of cell ChannelManager
  ServiceManagerPtr service_manager_;  ///< shared ptr of cell ServiceManager
  /// rtps participant for cell topology pub/sub broadcast
  transport::ParticipantPtr participant_;
  ParticipantListener* participant_listener_;
  /// Cell topology change signal; notifies all registered ChangeFunc observers
  /// [ASTRO-TOPO] debug: fires on every cell Join/Leave event
  ChangeSignal change_signal_;
  /// Known remote cell participants in the topology: GUID -> participant_name
  /// [ASTRO-TOPO] debug: used to reconstruct Leave events from GUID on dropout
  PartNameContainer participant_names_;

  DECLARE_SINGLETON(TopologyManager)
};

}  // namespace service_discovery
}  // namespace cyber
}  // namespace apollo

#endif  // CYBER_SERVICE_DISCOVERY_TOPOLOGY_MANAGER_H_
