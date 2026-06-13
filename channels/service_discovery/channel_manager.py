import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



class AstroChannelManager:
    """
    Astro 通道拓扑管理器 — Python port of ChannelManager。

    维护两组索引：
      _channel_writers / _channel_readers : Dict[channel_path, Dict[role_id, RoleRecord]]
      _node_writers    / _node_readers    : Dict[node_name,    Dict[role_id, RoleRecord]]

    以及有向图（node_graph_）和 OnTopoChange 回调列表。

    线程安全：所有结构修改持有 self._mu。
    """

    _instance: Optional["AstroChannelManager"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._mu = threading.Lock()

        # channel_id(str) → {role_id: RoleRecord}
        self._channel_writers: Dict[str, Dict[str, RoleRecord]] = {}
        self._channel_readers: Dict[str, Dict[str, RoleRecord]] = {}

        # node_name → {role_id: RoleRecord}
        self._node_writers: Dict[str, Dict[str, RoleRecord]] = {}
        self._node_readers: Dict[str, Dict[str, RoleRecord]] = {}

        # node_graph_: node_name → set of downstream node_names
        # 有向边：writer_node → reader_node，边 value = channel_path
        # 简化为 {src_node: {dst_node}} — 足够 HasWriter/HasReader 之上的上下游查询
        self._node_graph: Dict[str, set] = {}

        # 拓扑变化回调列表 — 对应 change_signal_
        self._topo_listeners: List[Callable[[ChangeEvent], None]] = []

        _chanmgr_log.debug("[ASTRO-CHANMGR] AstroChannelManager constructed")

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "AstroChannelManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    # ── topo-change signal ───────────────────────────────────────────────────

    def add_topo_listener(
        self, callback: Callable[[ChangeEvent], None]
    ) -> Callable[[ChangeEvent], None]:
        """
        AddChangeListener — 注册拓扑变化回调。
        返回 callback 本身，便于调用方持有引用后 remove。
        对应 TopologyManager::AddChangeListener / change_signal_.Connect()。
        """
        with self._mu:
            self._topo_listeners.append(callback)
        _chanmgr_log.debug("[ASTRO-CHANMGR] add_topo_listener registered")
        return callback

    def remove_topo_listener(
        self, callback: Callable[[ChangeEvent], None]
    ) -> None:
        """RemoveChangeListener — 注销拓扑变化回调。"""
        with self._mu:
            try:
                self._topo_listeners.remove(callback)
            except ValueError:
                pass

    def _notify(self, event: ChangeEvent) -> None:
        """
        Notify — 触发所有已注册回调。
        对应 Manager::Notify() → change_signal_(msg)。
        回调在锁外串行调用（避免重入死锁，同 Apollo Signal::operator()）。
        """
        with self._mu:
            cbs = list(self._topo_listeners)
        for cb in cbs:
            try:
                cb(event)
            except Exception as exc:  # noqa: BLE001
                _chanmgr_log.warning(
                    "[ASTRO-CHANMGR] topo_listener exception: %s", exc
                )

    # ── internal warehouse helpers ───────────────────────────────────────────

    @staticmethod
    def _make_role_id(node_name: str, channel_path: str) -> str:
        """确定性 role_id — 对应 proto.id = HashValue(node+channel)。"""
        return f"{node_name}::{channel_path}"

    def _add_role(
        self,
        warehouse_ch: Dict[str, Dict[str, RoleRecord]],
        warehouse_nd: Dict[str, Dict[str, RoleRecord]],
        role: RoleRecord,
    ) -> None:
        """MultiValueWarehouse::Add — 双索引写入（channel 维度 + node 维度）。"""
        # channel 维度
        warehouse_ch.setdefault(role.channel_path, {})[role.role_id] = role
        # node 维度
        warehouse_nd.setdefault(role.node_name, {})[role.role_id] = role

    def _remove_role(
        self,
        warehouse_ch: Dict[str, Dict[str, RoleRecord]],
        warehouse_nd: Dict[str, Dict[str, RoleRecord]],
        role: RoleRecord,
    ) -> None:
        """MultiValueWarehouse::Remove — 双索引删除。"""
        bucket_ch = warehouse_ch.get(role.channel_path)
        if bucket_ch:
            bucket_ch.pop(role.role_id, None)
            if not bucket_ch:
                del warehouse_ch[role.channel_path]

        bucket_nd = warehouse_nd.get(role.node_name)
        if bucket_nd:
            bucket_nd.pop(role.role_id, None)
            if not bucket_nd:
                del warehouse_nd[role.node_name]

    # ── graph helpers ────────────────────────────────────────────────────────

    def _graph_insert_writer(self, node_name: str) -> None:
        """node_graph_.Insert(e.set_src(v)) — 确保 writer 节点存在。"""
        self._node_graph.setdefault(node_name, set())

    def _graph_insert_reader(
        self, writer_node: str, reader_node: str
    ) -> None:
        """node_graph_.Insert(e.set_dst(v)) — 建立 writer→reader 有向边。"""
        self._node_graph.setdefault(writer_node, set()).add(reader_node)

    def _graph_delete_edge(
        self, node_a: str, node_b: str, as_writer: bool
    ) -> None:
        """node_graph_.Delete — 移除有向边或孤立顶点。"""
        if as_writer:
            bucket = self._node_graph.get(node_a)
            if bucket is not None:
                bucket.discard(node_b)
        # reader 离开：从所有 writer 节点的邻居集合中删除 reader_node
        else:
            for adj in self._node_graph.values():
                adj.discard(node_a)

    # ── Join / Leave (DisposeJoin / DisposeLeave) ───────────────────────────

    def join(
        self,
        channel_path: str,
        node_name: str,
        role_type: str,
        msg_type: str = "",
        host_name: str = "localhost",
        process_id: int = 0,
    ) -> None:
        """
        Join — 注册 Writer 或 Reader 到拓扑。
        对应 ChannelManager::DisposeJoin(msg)。

        Parameters
        ----------
        channel_path : str  通道路径（替代 channel_name + channel_id）
        node_name    : str  节点名称
        role_type    : str  "WRITER" | "READER"
        msg_type     : str  消息类型标识（可选）
        host_name    : str  主机名（用于 OnTopoModuleLeave）
        process_id   : int  进程 ID
        """
        role_id = self._make_role_id(node_name, channel_path)
        role = RoleRecord(
            node_name=node_name,
            channel_path=channel_path,
            role_id=role_id,
            msg_type=msg_type,
            timestamp=_time.monotonic(),
            host_name=host_name,
            process_id=process_id,
        )

        with self._mu:
            self._scan_message_type(channel_path, msg_type)

            if role_type == "WRITER":
                self._add_role(
                    self._channel_writers, self._node_writers, role
                )
                self._graph_insert_writer(node_name)
            else:
                self._add_role(
                    self._channel_readers, self._node_readers, role
                )
                # 建立所有当前 writer 节点 → reader 节点的有向边
                for wr in self._channel_writers.get(channel_path, {}).values():
                    self._graph_insert_reader(wr.node_name, node_name)

        _chanmgr_log.debug(
            "[ASTRO-CHANMGR] join ch=%s node=%s role=%s",
            channel_path, node_name, role_type,
        )
        event = ChangeEvent(
            operate_type="JOIN", role_type=role_type, role=role
        )
        self._notify(event)

    def leave(
        self,
        channel_path: str,
        node_name: str,
        role_type: str,
    ) -> None:
        """
        Leave — 从拓扑注销 Writer 或 Reader。
        对应 ChannelManager::DisposeLeave(msg)。
        """
        role_id = self._make_role_id(node_name, channel_path)
        role = RoleRecord(
            node_name=node_name,
            channel_path=channel_path,
            role_id=role_id,
        )

        with self._mu:
            if role_type == "WRITER":
                self._remove_role(
                    self._channel_writers, self._node_writers, role
                )
                self._graph_delete_edge(node_name, "", as_writer=True)
            else:
                self._remove_role(
                    self._channel_readers, self._node_readers, role
                )
                self._graph_delete_edge(node_name, "", as_writer=False)

        _chanmgr_log.debug(
            "[ASTRO-CHANMGR] leave ch=%s node=%s role=%s",
            channel_path, node_name, role_type,
        )
        event = ChangeEvent(
            operate_type="LEAVE", role_type=role_type, role=role
        )
        self._notify(event)

    # ── HasWriter / HasReader ────────────────────────────────────────────────

    def has_writer(self, channel_path: str) -> bool:
        """
        HasWriter — 查询通道是否有活跃 Writer。
        对应 ChannelManager::HasWriter(channel_name)。
        """
        with self._mu:
            return bool(self._channel_writers.get(channel_path))

    def has_reader(self, channel_path: str) -> bool:
        """
        HasReader — 查询通道是否有活跃 Reader。
        对应 ChannelManager::HasReader(channel_name)。
        """
        with self._mu:
            return bool(self._channel_readers.get(channel_path))

    # ── GetChannelNames ──────────────────────────────────────────────────────

    def get_channel_names(self) -> List[str]:
        """
        GetChannelNames — 返回拓扑中所有已知通道名称（去重）。
        对应 ChannelManager::GetChannelNames(std::vector<std::string>*)。

        原典：合并 channel_writers_ + channel_readers_ 中所有 role 的
        channel_name 到 unordered_set，再 move 到 vector。
        此处：两个 dict 的键本身已去重，取并集即可。
        """
        with self._mu:
            names: set = set(self._channel_writers.keys()) | set(
                self._channel_readers.keys()
            )
        result = sorted(names)
        _chanmgr_log.debug(
            "[ASTRO-CHANMGR] get_channel_names count=%d", len(result)
        )
        return result

    # ── GetWriters / GetReaders (按通道 / 按节点) ────────────────────────────

    def get_writers_of_channel(self, channel_path: str) -> List[RoleRecord]:
        """GetWritersOfChannel — 返回通道的所有 Writer RoleRecord 列表。"""
        with self._mu:
            bucket = self._channel_writers.get(channel_path, {})
            return list(bucket.values())

    def get_readers_of_channel(self, channel_path: str) -> List[RoleRecord]:
        """GetReadersOfChannel — 返回通道的所有 Reader RoleRecord 列表。"""
        with self._mu:
            bucket = self._channel_readers.get(channel_path, {})
            return list(bucket.values())

    def get_writers_of_node(self, node_name: str) -> List[RoleRecord]:
        """GetWritersOfNode — 返回节点的所有 Writer RoleRecord 列表。"""
        with self._mu:
            bucket = self._node_writers.get(node_name, {})
            return list(bucket.values())

    def get_readers_of_node(self, node_name: str) -> List[RoleRecord]:
        """GetReadersOfNode — 返回节点的所有 Reader RoleRecord 列表。"""
        with self._mu:
            bucket = self._node_readers.get(node_name, {})
            return list(bucket.values())

    # ── GetUpstream / GetDownstream ──────────────────────────────────────────

    def get_upstream_of_node(self, node_name: str) -> List[str]:
        """
        GetUpstreamOfNode — 返回 node_name 的上游节点名称列表。

        逻辑与 channel_manager.cc 相同：
          1. 找 node_name 订阅的所有 channel。
          2. 找这些 channel 的所有 writer 节点。
          原典返回 RoleAttrVec；此处只返回 node_name 字符串列表（更 Pythonic）。
        """
        with self._mu:
            readers = list(
                self._node_readers.get(node_name, {}).values()
            )
        channels = {r.channel_path for r in readers}
        upstreams: set = set()
        for ch in channels:
            for wr in self.get_writers_of_channel(ch):
                if wr.node_name != node_name:
                    upstreams.add(wr.node_name)
        return sorted(upstreams)

    def get_downstream_of_node(self, node_name: str) -> List[str]:
        """
        GetDownstreamOfNode — 返回 node_name 的下游节点名称列表。
        逻辑对称于 get_upstream_of_node。
        """
        with self._mu:
            writers = list(
                self._node_writers.get(node_name, {}).values()
            )
        channels = {w.channel_path for w in writers}
        downstreams: set = set()
        for ch in channels:
            for rd in self.get_readers_of_channel(ch):
                if rd.node_name != node_name:
                    downstreams.add(rd.node_name)
        return sorted(downstreams)

    # ── OnTopoModuleLeave ────────────────────────────────────────────────────

    def on_topo_module_leave(
        self, host_name: str, process_id: int
    ) -> None:
        """
        OnTopoModuleLeave — 批量清除某进程的所有 Writer/Reader。
        对应 ChannelManager::OnTopoModuleLeave(host_name, process_id)。

        原典：用 RoleAttributes{host_name, process_id} 为 key 批量 Search，
        再对每个匹配角色生成 DisposeLeave + Notify ChangeMsg。
        此处：扫描全表，按 role.host_name + role.process_id 前缀匹配。
        """
        _chanmgr_log.debug(
            "[ASTRO-CHANMGR] on_topo_module_leave host=%s pid=%d",
            host_name, process_id,
        )
        to_remove_writers: List[RoleRecord] = []
        to_remove_readers: List[RoleRecord] = []

        with self._mu:
            for bucket in self._channel_writers.values():
                for role in bucket.values():
                    if (
                        role.host_name == host_name
                        and role.process_id == process_id
                    ):
                        to_remove_writers.append(role)
            for bucket in self._channel_readers.values():
                for role in bucket.values():
                    if (
                        role.host_name == host_name
                        and role.process_id == process_id
                    ):
                        to_remove_readers.append(role)

        for role in to_remove_writers:
            self.leave(role.channel_path, role.node_name, "WRITER")
        for role in to_remove_readers:
            self.leave(role.channel_path, role.node_name, "READER")

    # ── ScanMessageType (internal) ───────────────────────────────────────────

    def _scan_message_type(
        self, channel_path: str, new_msg_type: str
    ) -> None:
        """
        ScanMessageType — 检查新角色的消息类型是否与已有 writer/reader 冲突。
        对应 ChannelManager::ScanMessageType(msg)。

        Apollo 原典在 AERROR 级别打印不匹配警告；此处改为 logging.warning。
        豁免类型（_EXEMPTED_MSG_TYPES）不做检查，与原典 exempted_msg_types_ 一致。
        注意：此方法在持锁状态下调用（join() 内部），不重复加锁。
        """
        if new_msg_type in _EXEMPTED_MSG_TYPES or not new_msg_type:
            return

        for wr in self._channel_writers.get(channel_path, {}).values():
            if (
                wr.msg_type
                and wr.msg_type not in _EXEMPTED_MSG_TYPES
                and wr.msg_type != new_msg_type
            ):
                _chanmgr_log.warning(
                    "[ASTRO-CHANMGR] msg_type mismatch ch=%s "
                    "existing_writer_type=%s new_type=%s",
                    channel_path, wr.msg_type, new_msg_type,
                )

        for rd in self._channel_readers.get(channel_path, {}).values():
            if (
                rd.msg_type
                and rd.msg_type not in _EXEMPTED_MSG_TYPES
                and rd.msg_type != new_msg_type
            ):
                _chanmgr_log.warning(
                    "[ASTRO-CHANMGR] msg_type mismatch ch=%s "
                    "existing_reader_type=%s new_type=%s",
                    channel_path, rd.msg_type, new_msg_type,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# AstroTopologyManager — ported from
#   upstream/apollo-cyber/service_discovery/topology_manager.h
#   upstream/apollo-cyber/service_discovery/topology_manager.cc
#
# 原典：TopologyManager 是单例，持有 NodeManager / ChannelManager /
# ServiceManager 三个子管理器，以及一个 DDS/RTPS Participant 用于跨进程
# 拓扑广播。change_signal_ 向所有注册的 ChangeFunc 广播拓扑变化事件。
# OnParticipantChange 在 RTPS Participant 发现/离开时触发，解析
# participant_name（"host_name+process_id"）后调用三个子管理器的
# OnTopoModuleLeave 并广播 ChangeMsg。
#
# 算法改动（20% 规则）：
#   1. RTPS Participant（eprosima fast-rtps）→ 无实际网络层；
#      跨进程发现由外部调用 notify_participant_leave() 模拟。
#   2. NodeManager / ServiceManager → 仅保留 AstroChannelManager；
#      节点/服务维度的拓扑管理通过 AstroChannelManager 间接覆盖。
#   3. DECLARE_SINGLETON → classmethod instance() + _instance。
#   4. ChangeConnection (base::Connection) → 返回 callback 本身（同 add_topo_listener）。
#   5. participant_names_ GUID→str map → _participant_names: dict[str, str]
#      key 改为 participant_name 字符串（进程内无 GUID）。
#   6. Convert(PartInfo) → _convert_participant_event(host_name, process_id, op)。
#   7. 新增 snapshot() — 返回当前拓扑快照 dict，供调试和测试使用。
#      （原典无此接口；TopologyManager 状态散布在三个子管理器中，快照需聚合。）
#   8. OnTopoChange 回调参数改为 ChangeEvent（Astro 本地类型）而非 protobuf ChangeMsg。
#
# Debug prefix: [ASTRO-TOPOMGR] — 对应 topology_manager.cc 中的 ADEBUG 前缀。
# ═══════════════════════════════════════════════════════════════════════════════



from channels.node import *




# ───────────────────────────────────────────────────────────────────────────────
# ChannelRegistry — lightweight topology map
#
# Mirrors the ChannelManager (service_discovery/channel_manager.h) role:
# tracks which writers and readers are currently active on each channel path.
# Apollo uses a distributed shared-memory AtomicHashMap keyed by channel_id
# (uint64_t).  Here we use a plain dict keyed by channel_path (string) guarded
# by a threading.Lock so the registry is safe for multi-threaded test scenarios.
# ───────────────────────────────────────────────────────────────────────────────

class ChannelRegistry:
    """
    Singleton topology registry.

    Maps channel_path → {writers: set[str], readers: set[str]}
    where each entry is the role_id (node_name + channel_path hash).

    Mirrors ChannelManager::Join / Leave / HasWriter / HasReader.
    """
    _instance: Optional["ChannelRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._map: Dict[str, Dict[str, set]] = {}   # path → {writers, readers}
        self._mu = threading.Lock()

    @classmethod
    def instance(cls) -> "ChannelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    def _ensure(self, channel_path: str):
        if channel_path not in self._map:
            self._map[channel_path] = {"writers": set(), "readers": set()}

    def join_writer(self, channel_path: str, role_id: str):
        with self._mu:
            self._ensure(channel_path)
            self._map[channel_path]["writers"].add(role_id)

    def leave_writer(self, channel_path: str, role_id: str):
        with self._mu:
            if channel_path in self._map:
                self._map[channel_path]["writers"].discard(role_id)

    def join_reader(self, channel_path: str, role_id: str):
        with self._mu:
            self._ensure(channel_path)
            self._map[channel_path]["readers"].add(role_id)

    def leave_reader(self, channel_path: str, role_id: str):
        with self._mu:
            if channel_path in self._map:
                self._map[channel_path]["readers"].discard(role_id)

    def has_writer(self, channel_path: str) -> bool:
        with self._mu:
            return bool(self._map.get(channel_path, {}).get("writers"))

    def has_reader(self, channel_path: str) -> bool:
        with self._mu:
            return bool(self._map.get(channel_path, {}).get("readers"))

    def writer_count(self, channel_path: str) -> int:
        with self._mu:
            return len(self._map.get(channel_path, {}).get("writers", set()))

    def reader_count(self, channel_path: str) -> int:
        with self._mu:
            return len(self._map.get(channel_path, {}).get("readers", set()))


# ───────────────────────────────────────────────────────────────────────────────
# AstroBlocker — dual-queue (publish_queue / observe_queue)
#
# Ports blocker::Blocker<MessageT> (cyber/blocker/blocker.h).
# Apollo Blocker keeps two std::list<shared_ptr<T>>:
#   published_ — messages arrive here (Publish); bounded by capacity.
#   observed_  — snapshots here on Observe(); drained by reader callbacks.
# ASTRO change: lists → collections.deque with maxlen for O(1) bounded append.
# ───────────────────────────────────────────────────────────────────────────────



# ═══════════════════════════════════════════════════════════════════════════════
# AstroListenerHandler — ported from
#   upstream/apollo-cyber/transport/message/listener_handler.h
#
# Original: template<MessageT> with Signal<shared_ptr<MessageT>, MessageInfo>
#   • Connect(self_id, listener) — single-cast slot (no oppo filter)
#   • Connect(self_id, oppo_id, listener) — filtered slot per sender
#   • Disconnect(self_id) / Disconnect(self_id, oppo_id)
#   • Run(msg, msg_info) — fires signal_ then per-oppo signals_[oppo_id]
#   • RunFromString(str, msg_info) — parse then Run (proto deserialise)
#
# ASTRO changes (20% algorithm delta):
#   1. MessageT template → duck-typed Python (any dict / str / object).
#   2. Signal<> + base::Connection<> → plain list of callables (simpler).
#   3. RunFromString: proto parse → json.loads (file-channel serialisation).
#   4. oppo_id: uint64 hash → str (channel path or role name).
#   5. AtomicRWLock → threading.Lock (Python GIL provides safety for reads;
#      lock only for structural mutations, matching the write-guard pattern).
#   6. is_raw_message_ flag preserved as _is_raw (bool).
# ═══════════════════════════════════════════════════════════════════════════════

import threading



@dataclasses.dataclass
class ChangeEvent:
    """
    拓扑变化事件 — 对应 apollo::cyber::proto::ChangeMsg。

    operate_type: "JOIN" | "LEAVE"
    role_type:    "WRITER" | "READER"
    role:         RoleRecord
    timestamp:    float (monotonic)
    """
    operate_type: str    # "JOIN" | "LEAVE"
    role_type: str       # "WRITER" | "READER"
    role: RoleRecord
    timestamp: float = dataclasses.field(default_factory=_time.monotonic)


# 豁免类型集合 — 对应 ExemptedMessageTypes
# Apollo 默认豁免 RawMessage / PyMessageWrap；此处以约定字符串替代 MessageType<T>()
_EXEMPTED_MSG_TYPES: frozenset = frozenset({"__raw__", "__py_wrap__"})



