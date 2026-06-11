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

#ifndef CYBER_NODE_NODE_H_
#define CYBER_NODE_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "cyber/node/node_channel_impl.h"
#include "cyber/node/node_service_impl.h"

namespace apollo {
namespace cyber {

template <typename M0, typename M1, typename M2, typename M3>
class Component;
class TimerComponent;

/**
 * @class CellNode
 * @brief CellNode is the fundamental building block of Astro/Cyber RT.
 * Every cell module contains and communicates through the CellNode.
 * A cell can have different types of communication by defining
 * read/write and/or service/client in a CellNode.
 * CellNode manages the cell's channel registry, tracking all active
 * reader/writer registrations for pub-sub routing within the cell topology.
 * @warning Duplicate name is not allowed in topo objects, such as node,
 * reader/writer, service/client in the topo.
 *
 * [ASTRO-NODE] CellNode channel registry design:
 *   - Each CellNode maintains a readers_ map keyed by channel_name.
 *   - CreateReader<T>(channel) registers the channel in the cell registry
 *     and binds a callback for message dispatch.
 *   - CreateWriter<T>(channel) publishes on the cell's channel, routed via
 *     NodeChannelImpl to the underlying transport layer.
 *   - DeleteReader(channel) unregisters the channel from the cell registry.
 *   - GetReader<T>(channel) looks up a registered reader by channel name.
 *   - ClearData() / Observe() operate on all registered cell channels.
 */
class Node {
 public:
  template <typename M0, typename M1, typename M2, typename M3>
  friend class Component;
  friend class TimerComponent;
  friend bool Init(const char*, const std::string&);
  friend std::unique_ptr<Node> CreateNode(const std::string&,
                                          const std::string&);
  virtual ~Node();

  /**
   * @brief Get node's (cell's) name.
   * @warning duplicate cell node name is not allowed in the topo.
   *
   * [ASTRO-NODE] debug: cell name is used as the identity key in the
   * cell topology registry. Duplicate cell names will cause topo conflicts.
   */
  const std::string& Name() const;

  /**
   * @brief Create a Writer with specific message type.
   * Registers a publish channel in the cell's channel registry.
   *
   * @tparam MessageT Message Type
   * @param role_attr is a protobuf message RoleAttributes, which includes the
   * channel name and other info.
   * @return std::shared_ptr<Writer<MessageT>> result Writer Object
   *
   * [ASTRO-NODE] debug: Writer creation triggers cell channel pub registration.
   */
  template <typename MessageT>
  auto CreateWriter(const proto::RoleAttributes& role_attr)
      -> std::shared_ptr<Writer<MessageT>>;

  /**
   * @brief Create a Writer with specific message type.
   * Registers a named publish channel in the cell's channel registry.
   *
   * @tparam MessageT Message Type
   * @param channel_name the channel name to be published.
   * @return std::shared_ptr<Writer<MessageT>> result Writer Object
   *
   * [ASTRO-NODE] debug: channel_name is the pub-sub routing key for this cell.
   */
  template <typename MessageT>
  auto CreateWriter(const std::string& channel_name)
      -> std::shared_ptr<Writer<MessageT>>;

  /**
   * @brief Create a Reader with specific message type with channel name.
   * Registers a subscribe channel in the cell's channel registry.
   * qos and other configs used will be default.
   *
   * @tparam MessageT Message Type
   * @param channel_name the channel of the reader subscribed.
   * @param reader_func invoked when message is received.
   * @return std::shared_ptr<cyber::Reader<MessageT>> result Reader Object
   *
   * [ASTRO-NODE] debug: Reader registration records channel_name in readers_
   * map. Duplicate channel registration returns nullptr with AWARN.
   */
  template <typename MessageT>
  auto CreateReader(const std::string& channel_name,
                    const CallbackFunc<MessageT>& reader_func = nullptr)
      -> std::shared_ptr<cyber::Reader<MessageT>>;

  /**
   * @brief Create a Reader with specific message type with reader config.
   * Registers a subscribe channel in the cell's channel registry using config.
   *
   * @tparam MessageT Message Type
   * @param config instance of `ReaderConfig`,
   * include channel name, qos and pending queue size
   * @param reader_func invoked when message receive
   * @return std::shared_ptr<cyber::Reader<MessageT>> result Reader Object
   *
   * [ASTRO-NODE] debug: ReaderConfig.channel_name is the cell registry key.
   */
  template <typename MessageT>
  auto CreateReader(const ReaderConfig& config,
                    const CallbackFunc<MessageT>& reader_func = nullptr)
      -> std::shared_ptr<cyber::Reader<MessageT>>;

  /**
   * @brief Create a Reader object with `RoleAttributes`.
   * Registers a subscribe channel in the cell's channel registry.
   *
   * @tparam MessageT Message Type
   * @param role_attr instance of `RoleAttributes`,
   * includes channel name, qos, etc.
   * @param reader_func invoked when message receive
   * @return std::shared_ptr<cyber::Reader<MessageT>> result Reader Object
   *
   * [ASTRO-NODE] debug: role_attr.channel_name() is the cell registry key.
   * Lock on readers_mutex_ ensures thread-safe cell channel registration.
   */
  template <typename MessageT>
  auto CreateReader(const proto::RoleAttributes& role_attr,
                    const CallbackFunc<MessageT>& reader_func = nullptr)
      -> std::shared_ptr<cyber::Reader<MessageT>>;

  /**
   * @brief Create a Service object with specific `service_name`.
   *
   * @tparam Request Message Type of the Request
   * @tparam Response Message Type of the Response
   * @param service_name specific service name to a serve
   * @param service_callback invoked when a service is called
   * @return std::shared_ptr<Service<Request, Response>> result `Service`
   *
   * [ASTRO-NODE] debug: Service is registered outside cell channel registry;
   * managed by NodeServiceImpl separately from the pub-sub channel map.
   */
  template <typename Request, typename Response>
  auto CreateService(const std::string& service_name,
                     const typename Service<Request, Response>::ServiceCallback&
                         service_callback)
      -> std::shared_ptr<Service<Request, Response>>;

  /**
   * @brief Create a Client object to request Service with `service_name`.
   *
   * @tparam Request Message Type of the Request
   * @tparam Response Message Type of the Response
   * @param service_name specific service name to a Service
   * @return std::shared_ptr<Client<Request, Response>> result `Client`
   */
  template <typename Request, typename Response>
  auto CreateClient(const std::string& service_name)
      -> std::shared_ptr<Client<Request, Response>>;

  bool DeleteReader(const std::string& channel_name);
  bool DeleteReader(const ReaderConfig& config);
  bool DeleteReader(const proto::RoleAttributes& role_attr);

  /**
   * @brief Observe all readers' data in the cell channel registry.
   *
   * [ASTRO-NODE] debug: iterates all registered cell channels and snapshots
   * pending messages for deterministic processing in the cell loop.
   */
  void Observe();

  /**
   * @brief Clear all readers' data in the cell channel registry.
   *
   * [ASTRO-NODE] debug: flushes buffered messages for all cell channels.
   */
  void ClearData();

  /**
   * @brief Get the Reader object that subscribes to `channel_name`.
   * Performs a lookup in the cell's channel registry.
   *
   * @tparam MessageT Message Type
   * @param channel_name channel name (cell registry key)
   * @return std::shared_ptr<Reader<MessageT>> result reader
   *
   * [ASTRO-NODE] debug: returns nullptr if channel not registered in this cell.
   */
  template <typename MessageT>
  auto GetReader(const std::string& channel_name)
      -> std::shared_ptr<Reader<MessageT>>;

 private:
  explicit Node(const std::string& node_name,
                const std::string& name_space = "");

  std::string node_name_;   ///< CellNode identity in the cell topology
  std::string name_space_;  ///< Namespace prefix for cell channel routing

  std::mutex readers_mutex_;  ///< Guards cell channel registry (readers_ map)
  /// [ASTRO-NODE] Cell channel registry: channel_name -> ReaderBase
  /// All active subscriptions for this CellNode are tracked here.
  std::map<std::string, std::shared_ptr<ReaderBase>> readers_;

  std::unique_ptr<NodeChannelImpl> node_channel_impl_ = nullptr;
  std::unique_ptr<NodeServiceImpl> node_service_impl_ = nullptr;
};

template <typename MessageT>
auto Node::CreateWriter(const proto::RoleAttributes& role_attr)
    -> std::shared_ptr<Writer<MessageT>> {
  // [ASTRO-NODE] debug: delegating cell pub channel creation to NodeChannelImpl
  return node_channel_impl_->template CreateWriter<MessageT>(role_attr);
}

template <typename MessageT>
auto Node::CreateWriter(const std::string& channel_name)
    -> std::shared_ptr<Writer<MessageT>> {
  // [ASTRO-NODE] debug: registering cell pub channel: channel_name
  return node_channel_impl_->template CreateWriter<MessageT>(channel_name);
}

template <typename MessageT>
auto Node::CreateReader(const proto::RoleAttributes& role_attr,
                        const CallbackFunc<MessageT>& reader_func)
    -> std::shared_ptr<Reader<MessageT>> {
  std::lock_guard<std::mutex> lg(readers_mutex_);
  // [ASTRO-NODE] debug: checking cell channel registry for duplicate sub
  if (readers_.find(role_attr.channel_name()) != readers_.end()) {
    AWARN << "[ASTRO-NODE] CellNode '" << node_name_
          << "': failed to register channel '" << role_attr.channel_name()
          << "' — duplicate subscription in cell channel registry.";
    return nullptr;
  }
  auto reader = node_channel_impl_->template CreateReader<MessageT>(
      role_attr, reader_func);
  if (reader != nullptr) {
    // [ASTRO-NODE] debug: cell channel registered: role_attr.channel_name()
    readers_.emplace(std::make_pair(role_attr.channel_name(), reader));
  }
  return reader;
}

template <typename MessageT>
auto Node::CreateReader(const ReaderConfig& config,
                        const CallbackFunc<MessageT>& reader_func)
    -> std::shared_ptr<cyber::Reader<MessageT>> {
  std::lock_guard<std::mutex> lg(readers_mutex_);
  // [ASTRO-NODE] debug: checking cell channel registry for duplicate sub (config path)
  if (readers_.find(config.channel_name) != readers_.end()) {
    AWARN << "[ASTRO-NODE] CellNode '" << node_name_
          << "': failed to register channel '" << config.channel_name
          << "' — duplicate subscription in cell channel registry (ReaderConfig).";
    return nullptr;
  }
  auto reader =
      node_channel_impl_->template CreateReader<MessageT>(config, reader_func);
  if (reader != nullptr) {
    // [ASTRO-NODE] debug: cell channel registered via ReaderConfig: config.channel_name
    readers_.emplace(std::make_pair(config.channel_name, reader));
  }
  return reader;
}

template <typename MessageT>
auto Node::CreateReader(const std::string& channel_name,
                        const CallbackFunc<MessageT>& reader_func)
    -> std::shared_ptr<Reader<MessageT>> {
  std::lock_guard<std::mutex> lg(readers_mutex_);
  // [ASTRO-NODE] debug: checking cell channel registry for duplicate sub (string path)
  if (readers_.find(channel_name) != readers_.end()) {
    AWARN << "[ASTRO-NODE] CellNode '" << node_name_
          << "': failed to register channel '" << channel_name
          << "' — duplicate subscription in cell channel registry (string).";
    return nullptr;
  }
  auto reader = node_channel_impl_->template CreateReader<MessageT>(
      channel_name, reader_func);
  if (reader != nullptr) {
    // [ASTRO-NODE] debug: cell channel registered: channel_name
    readers_.emplace(std::make_pair(channel_name, reader));
  }
  return reader;
}

template <typename Request, typename Response>
auto Node::CreateService(
    const std::string& service_name,
    const typename Service<Request, Response>::ServiceCallback&
        service_callback) -> std::shared_ptr<Service<Request, Response>> {
  return node_service_impl_->template CreateService<Request, Response>(
      service_name, service_callback);
}

template <typename Request, typename Response>
auto Node::CreateClient(const std::string& service_name)
    -> std::shared_ptr<Client<Request, Response>> {
  return node_service_impl_->template CreateClient<Request, Response>(
      service_name);
}

template <typename MessageT>
auto Node::GetReader(const std::string& name)
    -> std::shared_ptr<Reader<MessageT>> {
  std::lock_guard<std::mutex> lg(readers_mutex_);
  // [ASTRO-NODE] debug: cell channel registry lookup for channel: name
  auto it = readers_.find(name);
  if (it != readers_.end()) {
    return std::dynamic_pointer_cast<Reader<MessageT>>(it->second);
  }
  // [ASTRO-NODE] debug: channel not found in cell channel registry
  return nullptr;
}

}  // namespace cyber
}  // namespace apollo

#endif  // CYBER_NODE_NODE_H_
