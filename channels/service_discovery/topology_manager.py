import os, sys, json, threading, logging as _logging
from typing import Any, Callable, Dict, List, Optional, Generic, TypeVar
from channels.service_discovery.channel_manager import AstroChannelManager, ChangeEvent

_MT = TypeVar('_MT')
_chanmgr_log = _logging.getLogger("astro.topomgr")

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)


class AstroTopologyManager:
    """
    Astro 拓扑总管理器 — Python port of TopologyManager。

    持有 AstroChannelManager 单例，并提供：
      • OnTopoChange 回调注册 / 注销（对应 AddChangeListener / RemoveChangeListener）
      • 参与者 Join/Leave 事件派发（对应 OnParticipantChange）
      • 拓扑快照（snapshot()）

    Singleton: 通过 AstroTopologyManager.instance() 获取。

    Usage::

        topo = AstroTopologyManager.instance()
        conn = topo.add_change_listener(lambda ev: print(ev))

        topo.channel_manager.join("cell/out.json", "node_A", "WRITER")
        topo.channel_manager.join("cell/out.json", "node_B", "READER")

        snap = topo.snapshot()
        print(snap["channels"])          # {"cell/out.json": {writers:[…], readers:[…]}}
        print(snap["nodes"])             # {"node_A": {writers:[…], readers:[]}, …}

        topo.notify_participant_leave("worker-01", 12345)   # 模拟节点离线
        topo.remove_change_listener(conn)
    """

    _instance: Optional["AstroTopologyManager"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._init: bool = False
        self._mu = threading.Lock()

        # 唯一子管理器 — 对应 channel_manager_ shared_ptr
        self._channel_manager: AstroChannelManager = (
            AstroChannelManager.instance()
        )

        # 拓扑变化信号 — 对应 change_signal_
        # list of (callback, id) tuples；id 用于精确 remove
        self._listeners: List[Callable[[ChangeEvent], None]] = []

        # 已知参与者表 — 对应 participant_names_: GUID → participant_name
        # 此处以 participant_key(str) → bool 标记存活状态
        self._participant_names: Dict[str, str] = {}

        # 将 ChannelManager 的拓扑事件转发到 TopologyManager 的监听器
        self._channel_manager.add_topo_listener(self._on_channel_event)

        self._init = True
        _chanmgr_log.debug("[ASTRO-TOPOMGR] AstroTopologyManager constructed")

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "AstroTopologyManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Shutdown — 停止拓扑发现，断开所有回调。
        对应 TopologyManager::Shutdown()。
        """
        with self._mu:
            if not self._init:
                return
            self._init = False
            self._listeners.clear()
            self._participant_names.clear()
        _chanmgr_log.debug("[ASTRO-TOPOMGR] shutdown complete")

    def is_initialized(self) -> bool:
        with self._mu:
            return self._init

    # ── channel_manager accessor ─────────────────────────────────────────────

    @property
    def channel_manager(self) -> AstroChannelManager:
        """
        channel_manager() — 对应 TopologyManager::channel_manager()。
        返回 AstroChannelManager 单例引用。
        """
        return self._channel_manager

    # ── AddChangeListener / RemoveChangeListener ─────────────────────────────

    def add_change_listener(
        self, callback: Callable[[ChangeEvent], None]
    ) -> Callable[[ChangeEvent], None]:
        """
        AddChangeListener — 注册拓扑变化观察者。
        对应 TopologyManager::AddChangeListener(func) → ChangeConnection。

        返回 callback 自身作为 connection handle（同 remove_change_listener 参数）。
        """
        with self._mu:
            self._listeners.append(callback)
        _chanmgr_log.debug("[ASTRO-TOPOMGR] add_change_listener registered")
        return callback

    def remove_change_listener(
        self, callback: Callable[[ChangeEvent], None]
    ) -> None:
        """
        RemoveChangeListener — 注销拓扑变化观察者。
        对应 TopologyManager::RemoveChangeListener(conn)。
        """
        with self._mu:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

    def _fire_listeners(self, event: ChangeEvent) -> None:
        """
        change_signal_(msg) — 向所有监听器广播拓扑事件。
        在锁外调用，避免重入死锁（与 Apollo Signal::operator() 相同策略）。
        """
        with self._mu:
            cbs = list(self._listeners)
        for cb in cbs:
            try:
                cb(event)
            except Exception as exc:  # noqa: BLE001
                _chanmgr_log.warning(
                    "[ASTRO-TOPOMGR] change_listener exception: %s", exc
                )

    # ── OnTopoChange 内部转发 ────────────────────────────────────────────────

    def _on_channel_event(self, event: ChangeEvent) -> None:
        """
        OnTopoChange 内部回调 — 将 AstroChannelManager 的事件转发给
        AstroTopologyManager 的外部监听器。

        对应 TopologyManager::OnParticipantChange 调用 change_signal_(msg)
        的最后一步：无论是 JOIN 还是 LEAVE，都广播给所有注册观察者。

        [ASTRO-TOPOMGR] debug: 仅在 VERBOSE 模式下打印（避免性能影响）。
        """
        _chanmgr_log.debug(
            "[ASTRO-TOPOMGR] on_channel_event op=%s role=%s ch=%s node=%s",
            event.operate_type,
            event.role_type,
            event.role.channel_path,
            event.role.node_name,
        )
        if not self.is_initialized():
            return
        self._fire_listeners(event)

    # ── OnParticipantChange / notify_participant_leave ───────────────────────

    def notify_participant_join(
        self, host_name: str, process_id: int
    ) -> None:
        """
        模拟参与者加入 — 对应 OnParticipantChange(DISCOVERED_RTPSPARTICIPANT)。

        原典：从 PartInfo 解析 participant_name（"host+pid"），
        记入 participant_names_，广播 OPT_JOIN ChangeMsg。
        此处：直接接受 host_name + process_id，生成 JOIN ChangeEvent。
        """
        key = f"{host_name}+{process_id}"
        with self._mu:
            if not self._init:
                return
            self._participant_names[key] = key
        _chanmgr_log.debug(
            "[ASTRO-TOPOMGR] participant_join host=%s pid=%d",
            host_name, process_id,
        )
        # 构造一个虚拟 PARTICIPANT JOIN 事件广播给外部监听器
        synthetic_role = RoleRecord(
            node_name=f"{host_name}:{process_id}",
            channel_path="__participant__",
            role_id=key,
            host_name=host_name,
            process_id=process_id,
        )
        event = ChangeEvent(
            operate_type="JOIN",
            role_type="PARTICIPANT",
            role=synthetic_role,
        )
        self._fire_listeners(event)

    def notify_participant_leave(
        self, host_name: str, process_id: int
    ) -> None:
        """
        模拟参与者离开 — 对应 OnParticipantChange(REMOVED/DROPPED_RTPSPARTICIPANT)。

        原典：
          1. 从 participant_names_ 按 GUID 找到 participant_name。
          2. ParseParticipantName → host_name + process_id。
          3. 调用 node_manager_/channel_manager_/service_manager_
             的 OnTopoModuleLeave(host_name, process_id)。
          4. 广播 OPT_LEAVE ChangeMsg。

        此处：直接调用 channel_manager.on_topo_module_leave，
        然后广播合成 LEAVE 事件。
        """
        key = f"{host_name}+{process_id}"
        with self._mu:
            if not self._init:
                return
            self._participant_names.pop(key, None)

        _chanmgr_log.debug(
            "[ASTRO-TOPOMGR] participant_leave host=%s pid=%d — "
            "cleaning channel registrations",
            host_name, process_id,
        )
        # 通知 ChannelManager 清除该进程的所有角色
        self._channel_manager.on_topo_module_leave(host_name, process_id)

        # 广播合成 PARTICIPANT LEAVE 事件
        synthetic_role = RoleRecord(
            node_name=f"{host_name}:{process_id}",
            channel_path="__participant__",
            role_id=key,
            host_name=host_name,
            process_id=process_id,
        )
        event = ChangeEvent(
            operate_type="LEAVE",
            role_type="PARTICIPANT",
            role=synthetic_role,
        )
        self._fire_listeners(event)

    # ── snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """
        snapshot() — 返回当前拓扑的完整快照。

        原典（TopologyManager）无此接口；拓扑状态散布在三个子管理器中，
        调试时需分别查询。此处聚合为单一 dict，结构如下：

        {
          "channels": {
            "<channel_path>": {
              "writers": [{"node_name": …, "msg_type": …, "role_id": …}, …],
              "readers": [{"node_name": …, "msg_type": …, "role_id": …}, …],
            },
            …
          },
          "nodes": {
            "<node_name>": {
              "writers": [<channel_path>, …],
              "readers": [<channel_path>, …],
              "upstream":   [<node_name>, …],
              "downstream": [<node_name>, …],
            },
            …
          },
          "participants": [<"host+pid">, …],
          "snapshot_ts":  <float monotonic>,
        }

        用途：
          • 单元测试断言拓扑状态（无需逐个调用 has_writer/has_reader）。
          • 调试 / SVG 拓扑可视化的数据源。
        """
        ch_mgr = self._channel_manager

        channel_names = ch_mgr.get_channel_names()

        channels_snap: Dict[str, Any] = {}
        all_nodes: set = set()

        for ch in channel_names:
            writers = ch_mgr.get_writers_of_channel(ch)
            readers = ch_mgr.get_readers_of_channel(ch)
            channels_snap[ch] = {
                "writers": [
                    {
                        "node_name": r.node_name,
                        "msg_type": r.msg_type,
                        "role_id": r.role_id,
                        "host_name": r.host_name,
                        "process_id": r.process_id,
                    }
                    for r in writers
                ],
                "readers": [
                    {
                        "node_name": r.node_name,
                        "msg_type": r.msg_type,
                        "role_id": r.role_id,
                        "host_name": r.host_name,
                        "process_id": r.process_id,
                    }
                    for r in readers
                ],
            }
            for r in writers:
                all_nodes.add(r.node_name)
            for r in readers:
                all_nodes.add(r.node_name)

        nodes_snap: Dict[str, Any] = {}
        for node in sorted(all_nodes):
            wrs = ch_mgr.get_writers_of_node(node)
            rds = ch_mgr.get_readers_of_node(node)
            nodes_snap[node] = {
                "writers": sorted({r.channel_path for r in wrs}),
                "readers": sorted({r.channel_path for r in rds}),
                "upstream": ch_mgr.get_upstream_of_node(node),
                "downstream": ch_mgr.get_downstream_of_node(node),
            }

        with self._mu:
            participants = sorted(self._participant_names.keys())

        _chanmgr_log.debug(
            "[ASTRO-TOPOMGR] snapshot channels=%d nodes=%d participants=%d",
            len(channels_snap), len(nodes_snap), len(participants),
        )

        return {
            "channels": channels_snap,
            "nodes": nodes_snap,
            "participants": participants,
            "snapshot_ts": _time.monotonic(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AstroScheduler + AstroProcessor — ported from
#   upstream/apollo-cyber/scheduler/scheduler.h + scheduler.cc
#   upstream/apollo-cyber/scheduler/processor.h + processor.cc
#   upstream/apollo-cyber/scheduler/policy/scheduler_classic.h
#   upstream/apollo-cyber/scheduler/policy/classic_context.h
#
# Upstream originals (Apollo CyberRT classic policy):
#   Scheduler:
#     CreateTask(factory/func, name, visitor) — register CRoutine in dispatch table;
#       visitor != nullptr → RegisterNotifyCallback → NotifyProcessor on data arrive.
#     NotifyProcessor(crid) — wake the processor owning the coroutine.
#     DispatchTask(cr) — assign CRoutine to a ClassicContext MULTI_PRIO_QUEUE slot
#       keyed by (group_name, priority).
#     CheckSchedStatus() — per-processor utilisation snapshot.
#     Shutdown() — drain epoch slots; stop processor threads.
#   Processor:
#     Run() — spin loop: NextRoutine() → Resume() → Release(); Wait() on idle.
#     Stop() — set running_=false; join thread.
#     BindContext(ctx) — attach ProcessorContext; start thread via std::call_once.
#
# ASTRO changes (20% algorithm delta):
#   1. CRoutine (stackful coroutine)  → plain Python callable (func: () → None).
#   2. AtomicHashMap<uint64_t, …>     → dict[str, …] (task_name → entry).
#   3. MULTI_PRIO_QUEUE (array[20])   → heapq (z_layer as priority key).
#   4. ClassicContext.NextRoutine()   → AstroScheduler._dequeue_task() under lock.
#   5. Processor thread (std::thread) → ThreadPoolExecutor future.
#   6. crid (uint64_t hash)           → task_name (str), matching channel path style.
#   7. DataVisitorBase.RegisterNotifyCallback → on_message callback on channel_path.
#   8. epoch_index_ advance           → AstroScheduler.advance_epoch() (M126 port).
#   9. Snapshot struct                → AstroSnapshot dataclass (preserved fields).
#  10. CheckSchedStatus fprintf       → [ASTRO-SCHED] debug prefix (matches C++ tag).
#
# Debug prefix: [ASTRO-SCHED] / [ASTRO-PROC] — grep-compatible with C++ log tags.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Apollo Cyber Transport — Python Port
# Ported from upstream/apollo-cyber/transport/  (cell-pubsub-loop branch)
# Classes: AstroIdentity · AstroEndpoint · AstroMessageInfo
#          AstroHistory · AstroShmDispatcher · AstroRtpsDispatcher
#          AstroShmTransmitter · AstroHybridTransmitter
#          AstroShmReceiver  · AstroHybridReceiver
#
# 鲁迅曰：有谁从小康人家而坠入困顿的么，我以为在这途路中，大约可以看见世人
# 的真面目。——同理，一条消息从进程内直走到跨主机，沿途层层剥落的，正是那些
# 平日里藏得最深的锁、序号与身份。
# ═══════════════════════════════════════════════════════════════════════════════

import os
import struct
import threading
import time
import hashlib
import queue
import uuid
from collections import deque
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar
)

_MT = TypeVar("_MT")

# ── 内部调试 ──────────────────────────────────────────────────────────────────
_TRANSPORT_DEBUG = os.environ.get("ASTRO_TRANSPORT_DEBUG", "0") == "1"

def _tdbg(tag: str, msg: str) -> None:
    if _TRANSPORT_DEBUG:
        print(f"[{tag}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# AstroIdentity
# Port of: upstream/apollo-cyber/transport/common/identity.h
#
# 鲁迅曰：所谓身份，不过是八个字节的自欺欺人。哈希一算，人与人之间的区别
# 也不过如此。
# ══════════════════════════════════════════════════════════════════════════════

_ID_SIZE = 8  # constexpr uint8_t ID_SIZE = 8



