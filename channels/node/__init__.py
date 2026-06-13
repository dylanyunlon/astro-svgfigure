#!/usr/bin/env python3
"""
channels/node/__init__.py
Node-layer classes split from channel_runtime.py.
Covers: AstroBlocker, AstroCellReader, AstroCellWriter, ReaderConfig,
        AstroNodeChannelImpl, AstroReaderBase, AstroReceiverManager,
        WriterAttr, AstroWriterBase, AstroConcreteWriter,
        AstroService, AstroClient, AstroNodeServiceImpl.
"""

from __future__ import annotations

import abc
import collections
import concurrent.futures
import dataclasses as _dc
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Re-import runtime internals used by node classes
from channels.channel_runtime import (
    _dbg,
    DEFAULT_PENDING_QUEUE_SIZE,
    ChannelBuffer,
    ChannelRegistry,
    DataDispatcher,
    DataNotifier,
    Notifier,
    AstroChannelManager,
    RoleRecord,
)

class AstroBlocker:
    """
    Dual-queue message buffer for AstroCellReader.

    publish_queue  — hot side: new messages enqueued here by AstroCellWriter.
    observe_queue  — cold side: snapshot of publish_queue taken on observe().

    Mirrors blocker::Blocker<MessageT> with deque instead of std::list.
    """

    def __init__(self, channel_path: str, capacity: int = DEFAULT_PENDING_QUEUE_SIZE):
        self.channel_path = channel_path
        self._capacity = max(1, capacity)
        self._publish_queue: collections.deque = collections.deque(maxlen=self._capacity)
        self._observe_queue: collections.deque = collections.deque()
        self._mu = threading.Lock()

    # ── publish side ──────────────────────────────────────────────────────────

    def publish(self, msg: Any):
        """Publish — append to publish_queue; oldest drops when at capacity."""
        with self._mu:
            self._publish_queue.append(msg)

    def published_size(self) -> int:
        with self._mu:
            return len(self._publish_queue)

    def is_published_empty(self) -> bool:
        with self._mu:
            return len(self._publish_queue) == 0

    def clear_published(self):
        with self._mu:
            self._publish_queue.clear()

    # ── observe side ─────────────────────────────────────────────────────────

    def observe(self):
        """Observe — snapshot publish_queue into observe_queue. Mirrors Blocker::Observe."""
        with self._mu:
            self._observe_queue = collections.deque(self._publish_queue)

    def is_observed_empty(self) -> bool:
        with self._mu:
            return len(self._observe_queue) == 0

    def get_latest_observed(self) -> Optional[Any]:
        """GetLatestObservedPtr — rightmost element of observe_queue."""
        with self._mu:
            return self._observe_queue[-1] if self._observe_queue else None

    def get_oldest_observed(self) -> Optional[Any]:
        """GetOldestObservedPtr — leftmost element of observe_queue."""
        with self._mu:
            return self._observe_queue[0] if self._observe_queue else None

    def observed_list(self) -> List[Any]:
        """Return a snapshot list of the observe_queue (oldest → newest)."""
        with self._mu:
            return list(self._observe_queue)

    def clear_observed(self):
        with self._mu:
            self._observe_queue.clear()

    # ── capacity ─────────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self._capacity

    def set_capacity(self, depth: int):
        with self._mu:
            self._capacity = max(1, depth)
            new_q: collections.deque = collections.deque(self._publish_queue, maxlen=self._capacity)
            self._publish_queue = new_q


# ───────────────────────────────────────────────────────────────────────────────
# AstroCellReader — port of Reader<MessageT>
#
# Lifecycle mirrors Reader:
#   __init__  ≈ Reader(role_attr, reader_func, pending_queue_size)
#   init()    ≈ Reader::Init() — register with DataDispatcher + topology
#   shutdown()≈ Reader::Shutdown() — deregister
#   enqueue() ≈ Reader::Enqueue() — push msg, fire callback
#   observe() ≈ Reader::Observe() — snapshot publish → observe queue
#
# ASTRO algorithm delta:
#   • Coroutine / RoutineFactory → inline synchronous callback from enqueue().
#   • DataVisitor channel_id (uint64) → channel_path string.
#   • Blocker uses AstroBlocker (deque-backed) instead of std::list.
#   • JoinTheTopology → ChannelRegistry.join_reader.
#   • Timestamp uses time.monotonic() (wall-clock, no NTP jumps).
# ───────────────────────────────────────────────────────────────────────────────

class AstroCellReader:
    """
    Subscriber for a single channel path.

    Ports Reader<MessageT> from upstream/apollo-cyber/node/reader.h.

    Parameters
    ----------
    channel_path      : str      Channel path (replaces role_attr.channel_name).
    reader_func       : callable Optional callback(msg: dict) — fires on each enqueue.
    pending_queue_size: int      Blocker publish-queue depth (default 1).
    node_name         : str      Logical node name for topology registry.
    """

    def __init__(
        self,
        channel_path: str,
        reader_func: Optional[Callable[[Any], None]] = None,
        pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE,
        node_name: str = "astro_node",
    ):
        # role attributes (flat, no protobuf)
        self.channel_path = channel_path
        self.node_name = node_name
        self._pending_queue_size = pending_queue_size
        self._reader_func = reader_func

        # Blocker — mirrors BlockerPtr in Reader<MessageT>
        self._blocker = AstroBlocker(channel_path, capacity=pending_queue_size)

        # timing — mirrors latest_recv_time_sec_ / second_to_lastest_recv_time_sec_
        self._latest_recv_mono: float = -1.0
        self._second_to_latest_recv_mono: float = -1.0

        # init flag — mirrors std::atomic<bool> init_
        self._init: bool = False
        self._mu = threading.Lock()

        # topology role_id — deterministic string, mirrors role_attr.id (HashValue)
        self._role_id = f"{node_name}::{channel_path}"

        # channel buffer registered with DataDispatcher
        self._channel_buf: Optional[ChannelBuffer] = None

        _dbg("ASTRO-CHANNEL",
             f"AstroCellReader ctor channel={channel_path} node={node_name} "
             f"pending_q={pending_queue_size}")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def init(self) -> bool:
        """
        Init — mirrors Reader::Init().

        1. Creates a ChannelBuffer and registers it with DataDispatcher.
        2. Registers an on_message (DataNotifier) callback that calls enqueue().
        3. Joins ChannelRegistry as READER.
        """
        with self._mu:
            if self._init:
                return True

            # Register ChannelBuffer — mirrors ReceiverManager::GetReceiver
            self._channel_buf = ChannelBuffer(self.channel_path,
                                              max_size=self._pending_queue_size)
            DataDispatcher.instance().add_buffer(self._channel_buf)

            # Register notification callback — mirrors RoutineFactory / Scheduler::CreateTask.
            # When DataDispatcher.dispatch() fires DataNotifier.notify(), this callback
            # pulls the latest message from the buffer and calls enqueue().
            def _on_notify():
                msg = self._channel_buf.latest() if self._channel_buf else None
                if msg is not None:
                    self.enqueue(msg)

            DataNotifier.instance().add_notifier(
                self.channel_path, Notifier(_on_notify)
            )

            # Join topology
            ChannelRegistry.instance().join_reader(self.channel_path, self._role_id)

            self._init = True

        _dbg("ASTRO-CHANNEL",
             f"AstroCellReader init ok channel={self.channel_path} role={self._role_id}")
        return True

    def shutdown(self):
        """Shutdown — mirrors Reader::Shutdown(). Leaves topology; does not drain queue."""
        with self._mu:
            if not self._init:
                return
            self._init = False

        ChannelRegistry.instance().leave_reader(self.channel_path, self._role_id)
        _dbg("ASTRO-CHANNEL",
             f"AstroCellReader shutdown channel={self.channel_path}")

    # ── message flow ─────────────────────────────────────────────────────────

    def enqueue(self, msg: Any):
        """
        Enqueue — mirrors Reader::Enqueue().

        Updates timing accumulators, pushes msg to Blocker's publish_queue,
        fires reader_func callback if set.

        [ASTRO-CHANNEL] log mirrors the fprintf in reader.h Enqueue().
        """
        self._second_to_latest_recv_mono = self._latest_recv_mono
        self._latest_recv_mono = time.monotonic()

        seq = self._blocker.published_size()   # approx seq from publish depth
        ts_us = int(self._latest_recv_mono * 1e6)

        _dbg("ASTRO-CHANNEL",
             f"Reader enqueued message on channel '{self.channel_path}' "
             f"| seq={seq} | timestamp={ts_us}")

        self._blocker.publish(msg)

        if self._reader_func is not None:
            self._reader_func(msg)

    def observe(self):
        """
        Observe — mirrors Reader::Observe().

        Snapshots publish_queue into observe_queue so callers can iterate
        without racing with concurrent enqueue() calls.

        [ASTRO-CHANNEL] log mirrors the fprintf in reader.h Observe().
        """
        seq = self._blocker.published_size()
        ts_us = int(time.monotonic() * 1e6)
        _dbg("ASTRO-CHANNEL",
             f"Reader observed message on channel '{self.channel_path}' "
             f"| seq={seq} | timestamp={ts_us}")
        self._blocker.observe()

    # ── query interface ───────────────────────────────────────────────────────

    def has_received(self) -> bool:
        """HasReceived — publish_queue non-empty. Mirrors Reader::HasReceived."""
        return not self._blocker.is_published_empty()

    def empty(self) -> bool:
        """Empty — observe_queue is empty. Mirrors Reader::Empty."""
        return self._blocker.is_observed_empty()

    def get_delay_sec(self) -> float:
        """
        GetDelaySec — mirrors Reader::GetDelaySec().

        Returns max(age_since_last, inter_message_interval) so callers can
        detect both stale sources and bursty delivery.
        """
        if self._latest_recv_mono < 0:
            return -1.0
        now = time.monotonic()
        if self._second_to_latest_recv_mono < 0:
            return now - self._latest_recv_mono
        return max(
            now - self._latest_recv_mono,
            self._latest_recv_mono - self._second_to_latest_recv_mono,
        )

    def pending_queue_size(self) -> int:
        """PendingQueueSize — mirrors Reader::PendingQueueSize."""
        return self._pending_queue_size

    def get_latest_observed(self) -> Optional[Any]:
        """GetLatestObserved — mirrors Reader::GetLatestObserved."""
        return self._blocker.get_latest_observed()

    def get_oldest_observed(self) -> Optional[Any]:
        """GetOldestObserved — mirrors Reader::GetOldestObserved."""
        return self._blocker.get_oldest_observed()

    def clear_data(self):
        """ClearData — mirrors Reader::ClearData. Drains both queues."""
        self._blocker.clear_published()
        self._blocker.clear_observed()

    def set_history_depth(self, depth: int):
        """SetHistoryDepth — resize Blocker capacity. Mirrors Reader::SetHistoryDepth."""
        self._blocker.set_capacity(depth)

    def get_history_depth(self) -> int:
        """GetHistoryDepth — mirrors Reader::GetHistoryDepth."""
        return self._blocker.capacity

    def has_writer(self) -> bool:
        """HasWriter — query topology registry. Mirrors Reader::HasWriter."""
        if not self._init:
            return False
        return ChannelRegistry.instance().has_writer(self.channel_path)

    def is_init(self) -> bool:
        return self._init

    def get_channel_name(self) -> str:
        return self.channel_path


# ───────────────────────────────────────────────────────────────────────────────
# AstroCellWriter — port of Writer<MessageT>
#
# Lifecycle mirrors Writer:
#   __init__  ≈ Writer(role_attr)
#   init()    ≈ Writer::Init() — join topology
#   shutdown()≈ Writer::Shutdown()
#   write()   ≈ Writer::Write(msg) — dispatch + notify
#
# ASTRO algorithm delta:
#   • TransmitterPtr → DataDispatcher.dispatch() (file-backed JSON channel).
#   • protobuf serialise → json.dumps (JSON wire format).
#   • sizeof(MessageT) → len(json.dumps(msg)) for payload size logging.
#   • JoinTheTopology → ChannelRegistry.join_writer.
#   • AcquireMessage() → allocate_message() returning empty dict.
# ───────────────────────────────────────────────────────────────────────────────

class AstroCellWriter:
    """
    Publisher for a single channel path.

    Ports Writer<MessageT> from upstream/apollo-cyber/node/writer.h.

    Parameters
    ----------
    channel_path : str  Channel path (replaces role_attr.channel_name).
    node_name    : str  Logical node name for topology registry.
    """

    def __init__(self, channel_path: str, node_name: str = "astro_node"):
        self.channel_path = channel_path
        self.node_name = node_name

        # init flag — mirrors WriterBase bool init_ (mutex-guarded)
        self._init: bool = False
        self._mu = threading.Lock()

        # topology role_id
        self._role_id = f"{node_name}::{channel_path}"

        _dbg("ASTRO-CHANNEL",
             f"AstroCellWriter ctor channel={channel_path} node={node_name}")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def init(self) -> bool:
        """
        Init — mirrors Writer::Init().

        Registers with ChannelRegistry as WRITER so readers can discover
        this writer via AstroCellReader.has_writer().
        """
        with self._mu:
            if self._init:
                return True
            self._init = True

        ChannelRegistry.instance().join_writer(self.channel_path, self._role_id)
        _dbg("ASTRO-CHANNEL",
             f"AstroCellWriter init ok channel={self.channel_path} role={self._role_id}")
        return True

    def shutdown(self):
        """Shutdown — mirrors Writer::Shutdown. Leaves topology."""
        with self._mu:
            if not self._init:
                return
            self._init = False

        ChannelRegistry.instance().leave_writer(self.channel_path, self._role_id)
        _dbg("ASTRO-CHANNEL",
             f"AstroCellWriter shutdown channel={self.channel_path}")

    # ── publish ───────────────────────────────────────────────────────────────

    def write(self, msg: Any) -> bool:
        """
        Write — mirrors Writer::Write(msg).

        Serialises msg to JSON (replaces protobuf Transmit), routes through
        DataDispatcher (replaces transport::Transmitter::Transmit), which
        writes the file channel and fires DataNotifier callbacks.

        [ASTRO-CHANNEL] log mirrors the fprintf in writer.h Write().
        """
        if not self.is_init():
            return False

        # Approximate payload size — mirrors sizeof(MessageT) in writer.h log
        try:
            payload_bytes = len(json.dumps(msg).encode())
        except (TypeError, ValueError):
            payload_bytes = 0

        _dbg("ASTRO-CHANNEL",
             f"Writer published to channel '{self.channel_path}' "
             f"| size={payload_bytes} bytes")

        return DataDispatcher.instance().dispatch(self.channel_path, msg)

    def acquire_message(self) -> dict:
        """
        acquire_message — mirrors Writer::AcquireMessage().

        Returns a fresh empty dict; callers mutate it then pass to write().
        AcquireMessage in Apollo pre-allocates from a shared-memory pool;
        here we simply return {} since Python heap allocation is trivially cheap.
        """
        return {}

    # ── query interface ───────────────────────────────────────────────────────

    def has_reader(self) -> bool:
        """HasReader — query topology registry. Mirrors Writer::HasReader."""
        if not self.is_init():
            return False
        return ChannelRegistry.instance().has_reader(self.channel_path)

    def is_init(self) -> bool:
        with self._mu:
            return self._init

    def get_channel_name(self) -> str:
        return self.channel_path


# ───────────────────────────────────────────────────────────────────────────────
# Convenience factory — mirrors Node::CreateReader / Node::CreateWriter
# ───────────────────────────────────────────────────────────────────────────────

def create_reader(
    channel_path: str,
    reader_func: Optional[Callable[[Any], None]] = None,
    pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE,
    node_name: str = "astro_node",
    auto_init: bool = True,
) -> AstroCellReader:
    """
    create_reader — factory mirroring Node::CreateReader<MessageT>().

    Creates and optionally inits an AstroCellReader in one call.
    """
    r = AstroCellReader(channel_path, reader_func, pending_queue_size, node_name)
    if auto_init:
        r.init()
    return r


def create_writer(
    channel_path: str,
    node_name: str = "astro_node",
    auto_init: bool = True,
) -> AstroCellWriter:
    """
    create_writer — factory mirroring Node::CreateWriter<MessageT>().

    Creates and optionally inits an AstroCellWriter in one call.
    """
    w = AstroCellWriter(channel_path, node_name)
    if auto_init:
        w.init()
    return w
class ReaderConfig:
    """
    ReaderConfig — mirrors ``apollo::cyber::ReaderConfig``.

    Bundles channel_name + qos_depth + pending_queue_size.
    QosProfile fields not used by the Astro file-channel model are omitted;
    only the depth (HISTORY_KEEP_LAST) and pending_queue_size survive.
    """
    channel_name:      str = ""
    qos_depth:         int = 1          # QosProfile.depth default
    pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE


class AstroNodeChannelImpl:
    """
    Node channel factory — Python port of ``NodeChannelImpl``.

    Creates and lifecycle-manages AstroCellReader / AstroCellWriter pairs
    for one logical node, keeping the same factory semantics as the C++ Impl:

      • create_writer(channel_name)          → AstroCellWriter (init'd)
      • create_reader_by_name(ch, cb)        → AstroCellReader (init'd)
      • create_reader_by_config(cfg, cb)     → AstroCellReader (init'd)
      • create_reader_by_attr(cfg, cb, size) → AstroCellReader (init'd, explicit queue)

    All created objects are also registered with AstroChannelManager so the
    topology snapshot reflects the node's pub/sub graph.

    ASTRO delta from NodeChannelImpl::FillInAttr<M>:
      channel_id is computed as md5(channel_name)[:8].hex() — a deterministic
      8-byte string hash matching GlobalData::RegisterChannel semantics
      without requiring the Apollo globaldata singleton.
    """

    def __init__(
        self,
        node_name: str,
        mode: str = "INTRA",
        host_name: str = "localhost",
        process_id: int = 0,
    ) -> None:
        self.node_name:   str = node_name
        self._mode:       str = mode
        self._host_name:  str = host_name
        self._process_id: int = process_id

        # Join node-level topology — mirrors NodeManager::Join(node_attr_, ROLE_NODE)
        self._ch_mgr: AstroChannelManager = AstroChannelManager.instance()

        # Track created objects so __del__ can leave topology
        self._writers: List[AstroCellWriter] = []
        self._readers: List[AstroCellReader] = []

        _dbg("ASTRO-NODE-CH",
             f"NodeChannelImpl ctor node={node_name} mode={mode}")

    # ── internal attr helper ──────────────────────────────────────────────────

    @staticmethod
    def _register_channel(channel_name: str) -> str:
        """
        FillInAttr<M> channel_id assignment.

        Apollo: uint64_t channel_id = GlobalData::RegisterChannel(channel_name).
        ASTRO:  md5(channel_name)[:8].hex() — deterministic 16-hex-char string.
        """
        import hashlib as _hl
        return _hl.md5(channel_name.encode()).hexdigest()[:16]

    # ── CreateWriter ──────────────────────────────────────────────────────────

    def create_writer(self, channel_name: str) -> Optional[AstroCellWriter]:
        """
        CreateWriter(channel_name) — mirrors NodeChannelImpl::CreateWriter<M>
        (string overload).

        Constructs, inits, and topology-registers an AstroCellWriter.
        Returns None if channel_name is empty (mirrors the AERROR guard).
        """
        if not channel_name:
            _dbg("ASTRO-NODE-CH",
                 "create_writer: empty channel_name — abort")
            return None

        w = AstroCellWriter(channel_name, node_name=self.node_name)
        if not w.init():
            return None

        # Join topology — mirrors FillInAttr + NodeManager topology registration
        self._ch_mgr.join(
            channel_path=channel_name,
            node_name=self.node_name,
            role_type="WRITER",
            host_name=self._host_name,
            process_id=self._process_id,
        )

        self._writers.append(w)
        _dbg("ASTRO-NODE-CH",
             f"create_writer ok node={self.node_name} ch={channel_name} "
             f"ch_id={self._register_channel(channel_name)}")
        return w

    # ── CreateReader overloads ────────────────────────────────────────────────

    def create_reader_by_name(
        self,
        channel_name: str,
        reader_func: Optional[Callable[[Any], None]] = None,
    ) -> Optional[AstroCellReader]:
        """
        CreateReader(channel_name, reader_func) — string overload.

        Mirrors ``NodeChannelImpl::CreateReader<M>(channel_name, reader_func)``.
        pending_queue_size defaults to DEFAULT_PENDING_QUEUE_SIZE.
        """
        if not channel_name:
            return None
        cfg = ReaderConfig(channel_name=channel_name)
        return self.create_reader_by_config(cfg, reader_func)

    def create_reader_by_config(
        self,
        config: ReaderConfig,
        reader_func: Optional[Callable[[Any], None]] = None,
    ) -> Optional[AstroCellReader]:
        """
        CreateReader(config, reader_func) — ReaderConfig overload.

        Mirrors ``NodeChannelImpl::CreateReader<M>(ReaderConfig, reader_func)``.
        Extracts pending_queue_size from config.
        """
        return self.create_reader_by_attr(
            config, reader_func, config.pending_queue_size
        )

    def create_reader_by_attr(
        self,
        config: ReaderConfig,
        reader_func: Optional[Callable[[Any], None]] = None,
        pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE,
    ) -> Optional[AstroCellReader]:
        """
        CreateReader(role_attr, reader_func, pending_queue_size) — full overload.

        Mirrors the primary ``NodeChannelImpl::CreateReader<M>`` template.

        ASTRO path (always INTRA-mode):
          • Creates AstroCellReader with pending_queue_size.
          • Calls reader.init() — registers ChannelBuffer + DataNotifier callback.
          • Joins AstroChannelManager as READER for topology tracking.
        """
        channel_name = config.channel_name
        if not channel_name:
            _dbg("ASTRO-NODE-CH",
                 "create_reader_by_attr: empty channel_name — abort")
            return None

        r = AstroCellReader(
            channel_path=channel_name,
            reader_func=reader_func,
            pending_queue_size=pending_queue_size,
            node_name=self.node_name,
        )
        if not r.init():
            return None

        # Join topology — mirrors FillInAttr + ReceiverManager topology join
        self._ch_mgr.join(
            channel_path=channel_name,
            node_name=self.node_name,
            role_type="READER",
            host_name=self._host_name,
            process_id=self._process_id,
        )

        self._readers.append(r)
        _dbg("ASTRO-NODE-CH",
             f"create_reader ok node={self.node_name} ch={channel_name} "
             f"pending_q={pending_queue_size} "
             f"ch_id={self._register_channel(channel_name)}")
        return r

    # ── NodeName ──────────────────────────────────────────────────────────────

    def node_name_str(self) -> str:
        """NodeName() — const std::string& accessor."""
        return self.node_name

    # ── shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Leave topology for all created writers and readers.

        Mirrors NodeChannelImpl destructor:
          node_manager_->Leave(node_attr_, RoleType::ROLE_NODE)
        augmented with per-channel Leave so ChannelManager is consistent.
        """
        for w in self._writers:
            self._ch_mgr.leave(w.channel_path, self.node_name, "WRITER")
            w.shutdown()
        for r in self._readers:
            self._ch_mgr.leave(r.channel_path, self.node_name, "READER")
            r.shutdown()
        self._writers.clear()
        self._readers.clear()
        _dbg("ASTRO-NODE-CH",
             f"NodeChannelImpl shutdown node={self.node_name}")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# AstroReaderBase — ported from
#   upstream/apollo-cyber/node/reader_base.h
#
# 原典：ReaderBase 是所有 Reader<MessageT> 的抽象基类，持有 role_attr_
# （protobuf RoleAttributes）和 init_（std::atomic<bool>）；定义八个纯虚
# 接口（Init / Shutdown / ClearData / Observe / Empty / HasReceived /
# GetDelaySec / PendingQueueSize）以及若干非虚查询（GetChannelName /
# ChannelId / QosProfile / IsInit）。
#
# 同文件还定义了 ReceiverManager<MessageT> 单例——负责按 channel_name 去重，
# 保证每个 channel 只创建一个 Receiver；dispatch 回调经由 DataDispatcher<M>
# 把消息推入 ChannelBuffer，再由 DataNotifier 唤醒 DataVisitor。
#
# 算法改动（20% 规则）：
#   1. template<MessageT> ReaderBase → 非模板抽象基类（Python 无模板）。
#   2. proto::RoleAttributes → ReaderConfig dataclass（已在本文件定义）。
#   3. std::atomic<bool> init_ → threading.Event（init_set/init_clear）。
#   4. uint64_t channel_id → str channel_path（文件通道路径）。
#   5. QosProfile proto 字段 → qos_depth int（只保留 HISTORY_KEEP_LAST depth）。
#   6. ReceiverManager<M> 单例 → AstroReceiverManager 无泛型单例，
#      用 dict[channel_path, AstroCellReader] 保证去重。
#   7. DataDispatcher<M>::Dispatch + DataNotifier::Notify 回调由
#      AstroCellReader.init() 已完成注册，ReceiverManager 仅需返回 reader。
#
# 鲁迅曰：基类的存在，是为了让子类可以各行其是，却仍被同一把钥匙锁住——
# 这便是纯虚函数的权威，也是接口的温柔暴力。
#
# Debug prefix: [ASTRO-RBASE] — 与 C++ ADEBUG 前缀 grep 一致。
# ═══════════════════════════════════════════════════════════════════════════════

import abc


class AstroReaderBase(abc.ABC):
    """
    Abstract base class for all Astro channel readers.

    Ports ``apollo::cyber::ReaderBase`` from reader_base.h.

    Subclasses must implement the eight pure-virtual interfaces.
    The non-virtual accessors (get_channel_name, channel_id, is_init)
    are provided here using ReaderConfig and a threading.Event init flag.

    ASTRO delta: role_attr_ proto → ReaderConfig; init_ atomic → Event.
    """

    def __init__(self, config: ReaderConfig) -> None:
        self._config: ReaderConfig = config
        # std::atomic<bool> init_ — True once Init() succeeds
        self._init_event: threading.Event = threading.Event()

    # ── pure virtual interface ────────────────────────────────────────────────

    @abc.abstractmethod
    def init(self) -> bool:
        """Init() — initialise receiver; return True on success."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Shutdown() — deregister; release resources."""

    @abc.abstractmethod
    def clear_data(self) -> None:
        """ClearData() — drain all buffered messages."""

    @abc.abstractmethod
    def observe(self) -> None:
        """Observe() — snapshot publish-queue into observe-queue."""

    @abc.abstractmethod
    def empty(self) -> bool:
        """Empty() — True when observe-queue is empty."""

    @abc.abstractmethod
    def has_received(self) -> bool:
        """HasReceived() — True if any message arrived since last clear."""

    @abc.abstractmethod
    def get_delay_sec(self) -> float:
        """
        GetDelaySec() — seconds since last receive.

        Returns -1.0 before first message (mirrors Apollo convention).
        """

    @abc.abstractmethod
    def pending_queue_size(self) -> int:
        """PendingQueueSize() — configured depth of the pending queue."""

    # ── non-virtual accessors (mirror ReaderBase non-virtual methods) ─────────

    def has_writer(self) -> bool:
        """
        HasWriter() — query topology for an active writer.

        Default returns False (mirrors base class default body).
        Concrete subclasses override via ChannelRegistry.
        """
        return False

    def get_channel_name(self) -> str:
        """GetChannelName() — return role_attr_.channel_name()."""
        return self._config.channel_name

    def channel_id(self) -> str:
        """
        ChannelId() — deterministic string hash of channel_name.

        Apollo original: uint64_t from GlobalData::RegisterChannel.
        ASTRO: md5(channel_name)[:16] hex string.
        """
        import hashlib as _hl
        return _hl.md5(self._config.channel_name.encode()).hexdigest()[:16]

    def qos_depth(self) -> int:
        """QosProfile() accessor — return configured qos_depth."""
        return self._config.qos_depth

    def is_init(self) -> bool:
        """IsInit() — mirrors init_.load() atomic read."""
        return self._init_event.is_set()

    # ── init flag helpers (used by concrete subclasses) ───────────────────────

    def _mark_init(self) -> None:
        """Set init flag — call at end of successful init()."""
        self._init_event.set()
        _dbg("ASTRO-RBASE",
             f"init ok ch={self._config.channel_name}")

    def _clear_init(self) -> None:
        """Clear init flag — call in shutdown()."""
        self._init_event.clear()


# ── AstroReceiverManager — port of ReceiverManager<MessageT> singleton ────────

class AstroReceiverManager:
    """
    Per-channel reader de-duplicator — Python port of ReceiverManager<MessageT>.

    Apollo original: singleton template with AtomicHashMap<channel_name, Receiver>
    ensuring that multiple readers on the same channel share one Receiver so
    the DataDispatcher::Dispatch callback fires exactly once per message.

    ASTRO mapping:
      Receiver<M>           → AstroCellReader (already registered with dispatcher)
      receiver_map_[name]   → _map[channel_path]
      GetReceiver(role_attr) → get_or_create(channel_path, node_name, reader_func)

    Returns the existing AstroCellReader if already created for a channel,
    creating and init-ing a new one otherwise.
    """

    _instance: Optional["AstroReceiverManager"] = None
    _inst_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._map: Dict[str, AstroCellReader] = {}
        self._mu:  threading.Lock             = threading.Lock()

    @classmethod
    def instance(cls) -> "AstroReceiverManager":
        """DECLARE_SINGLETON equivalent."""
        if cls._instance is None:
            with cls._inst_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._inst_lock:
            cls._instance = None

    def get_receiver(
        self,
        channel_path: str,
        node_name: str = "astro_node",
        reader_func: Optional[Callable[[Any], None]] = None,
        pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE,
    ) -> Optional[AstroCellReader]:
        """
        GetReceiver(role_attr) — return shared reader for channel.

        Mirrors ReceiverManager<M>::GetReceiver: if a reader already exists
        for *channel_path* it is returned as-is (receiver_map_ de-dup).
        Otherwise a new AstroCellReader is created, init'd, and cached.

        reader_func and pending_queue_size are only used on first creation;
        subsequent calls with the same channel_path ignore them.
        """
        with self._mu:
            if channel_path in self._map:
                _dbg("ASTRO-RBASE",
                     f"ReceiverManager: reuse ch={channel_path}")
                return self._map[channel_path]

            r = AstroCellReader(
                channel_path=channel_path,
                reader_func=reader_func,
                pending_queue_size=pending_queue_size,
                node_name=node_name,
            )
            if not r.init():
                _dbg("ASTRO-RBASE",
                     f"ReceiverManager: init failed ch={channel_path}")
                return None

            self._map[channel_path] = r
            _dbg("ASTRO-RBASE",
                 f"ReceiverManager: new receiver ch={channel_path} node={node_name}")
            return r


# ═══════════════════════════════════════════════════════════════════════════════
# AstroWriterBase — ported from
#   upstream/apollo-cyber/node/writer_base.h
#
# 原典：WriterBase 是所有 Writer<MessageT> 的抽象基类，持有 role_attr_
# （proto::RoleAttributes）、init_ bool（std::mutex 保护）；
# 定义三个纯虚接口（Init / Shutdown / HasReader + GetReaders 非纯虚）以及
# 非虚查询（GetChannelName / GetChannelId / IsInit）。
# WriterBase 刻意不含消息发布接口（Write/Transmit）——这些留给 Writer<M> 模板。
#
# 算法改动（20% 规则）：
#   1. proto::RoleAttributes → WriterAttr dataclass（channel_name + node_name）。
#   2. std::mutex + bool init_ → threading.Lock + _init bool（语义完全一致）。
#   3. GetChannelId uint64_t → str hash（md5(channel_name)[:16]，与 ReaderBase 对称）。
#   4. HasReader / GetReaders 非纯虚默认实现 → 委托 ChannelRegistry（已在本文件）。
#   5. 无消息发布接口（同 C++ 设计意图）——发布由 AstroCellWriter.write() 提供。
#
# 鲁迅曰：写者无需知道读者在哪里——它只管写，有没有人读，是 HasReader 的事。
# 这正是发布-订阅的解耦哲学：一方沉默，另一方喧哗，互不相欠。
#
# Debug prefix: [ASTRO-WBASE] — 与 C++ ADEBUG 前缀 grep 一致。
# ═══════════════════════════════════════════════════════════════════════════════

@_dc.dataclass
class WriterAttr:
    """
    Lightweight writer role attributes — mirrors proto::RoleAttributes
    fields used by WriterBase (channel_name, node_name only).
    """
    channel_name: str = ""
    node_name:    str = "astro_node"


class AstroWriterBase(abc.ABC):
    """
    Abstract base class for all Astro channel writers.

    Ports ``apollo::cyber::WriterBase`` from writer_base.h.

    Subclasses must implement Init() and Shutdown().
    HasReader / GetReaders have default non-abstract implementations
    (mirrors C++ non-pure-virtual defaults).

    ASTRO delta:
      role_attr_ proto     → WriterAttr dataclass.
      std::mutex + bool    → threading.Lock + _init bool.
      GetChannelId uint64  → md5 hex string.
    """

    def __init__(self, attr: WriterAttr) -> None:
        self._attr: WriterAttr   = attr
        self._init: bool         = False
        self._lock: threading.Lock = threading.Lock()

    # ── pure virtual ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    def init(self) -> bool:
        """Init() — set up transmitter; return True on success."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Shutdown() — tear down transmitter; leave topology."""

    # ── non-virtual (mirrored from C++ non-pure-virtual defaults) ────────────

    def has_reader(self) -> bool:
        """
        HasReader() — query ChannelRegistry for active readers.

        Default returns False when not init'd (mirrors C++ default body).
        """
        if not self.is_init():
            return False
        return ChannelRegistry.instance().has_reader(self._attr.channel_name)

    def get_readers(self) -> List[str]:
        """
        GetReaders(readers*) — return list of reader role_ids on the channel.

        Mirrors WriterBase::GetReaders (vector<RoleAttributes>* out-param).
        ASTRO: returns role_id strings from ChannelRegistry.
        """
        with ChannelRegistry.instance()._mu:
            bucket = (
                ChannelRegistry.instance()
                ._map.get(self._attr.channel_name, {})
                .get("readers", set())
            )
            return sorted(bucket)

    # ── non-virtual accessors ─────────────────────────────────────────────────

    def get_channel_name(self) -> str:
        """GetChannelName() — const std::string& accessor."""
        return self._attr.channel_name

    def get_channel_id(self) -> str:
        """
        GetChannelId() — deterministic hex string channel identifier.

        Apollo: uint64_t from role_attr_.channel_id().
        ASTRO:  md5(channel_name)[:16].
        """
        import hashlib as _hl
        return _hl.md5(self._attr.channel_name.encode()).hexdigest()[:16]

    def is_init(self) -> bool:
        """IsInit() — mutex-guarded read of init_ bool."""
        with self._lock:
            return self._init

    # ── init flag helpers ─────────────────────────────────────────────────────

    def _mark_init(self) -> None:
        """Set init_ under lock — call at end of successful init()."""
        with self._lock:
            self._init = True
        _dbg("ASTRO-WBASE",
             f"init ok ch={self._attr.channel_name} node={self._attr.node_name}")

    def _clear_init(self) -> None:
        """Clear init_ under lock — call in shutdown()."""
        with self._lock:
            self._init = False


# ── Concrete minimal writer backed by AstroCellWriter ─────────────────────────

class AstroConcreteWriter(AstroWriterBase):
    """
    Concrete WriterBase subclass backed by AstroCellWriter.

    Bridges the abstract AstroWriterBase interface with the already-implemented
    AstroCellWriter pub/sub machinery, so callers that depend on the base-class
    interface (e.g. WriterBase::IsInit / HasReader / GetChannelId) work without
    duplicating implementation.

    write(msg) delegates directly to AstroCellWriter.write(msg).
    """

    def __init__(self, channel_name: str, node_name: str = "astro_node") -> None:
        super().__init__(WriterAttr(channel_name=channel_name, node_name=node_name))
        self._impl: AstroCellWriter = AstroCellWriter(channel_name, node_name)

    def init(self) -> bool:
        ok = self._impl.init()
        if ok:
            self._mark_init()
        return ok

    def shutdown(self) -> None:
        self._impl.shutdown()
        self._clear_init()

    def write(self, msg: Any) -> bool:
        """Write(msg) — delegates to AstroCellWriter.write()."""
        return self._impl.write(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# AstroNodeServiceImpl — ported from
#   upstream/apollo-cyber/node/node_service_impl.h
#
# 原典：NodeServiceImpl 是 Node 的私有 pImpl，持有 node_name_、
# RoleAttributes attr_（host_name / process_id / node_name / node_id）、
# service_list_（weak_ptr<ServiceBase> vector）和 client_list_（同上）；
# CreateService<Req,Resp> 构造 Service<Req,Resp>，调用 Init()，
# 注册到 TopologyManager::service_manager()（ROLE_SERVER）；
# CreateClient<Req,Resp> 构造 Client<Req,Resp>，调用 Init()，
# 注册到 TopologyManager::service_manager()（ROLE_CLIENT）。
#
# Astro 中没有独立的 Service / Client 模板（服务通信不在本文件的通道模型内），
# 因此我们用可调用的 Python handler 替代 ServiceCallback，
# 用 Future / threading.Event 替代 Client 的同步等待机制。
#
# 算法改动（20% 规则）：
#   1. template<Request, Response>    → handler: Callable[[Any], Any]（无泛型）。
#   2. Service<Req,Resp> + Init()     → AstroService（内嵌类，无 RTPS 层）。
#   3. Client<Req,Resp> + Init()      → AstroClient（threading.Event + result slot）。
#   4. TopologyManager service_manager → AstroChannelManager（以 service_name 为
#      channel_path，role_type = "WRITER"/"READER" 代替 ROLE_SERVER/ROLE_CLIENT）。
#   5. service_list_ / client_list_ weak_ptr 引用 → 强引用 list（Python GC 无需弱引用）。
#   6. GlobalData::RegisterService uint64_t → md5(service_name)[:16] hex string。
#   7. NodeServiceImpl() = delete      → __init__ 需要 node_name（无默认构造）。
#
# 鲁迅曰：服务与客户端，不过是两个在电话两端互望的人——一个等着被叫，一个拨出去
# 等结果。协议写得再严密，也逃不过那句\"请稍候\"。
#
# Debug prefix: [ASTRO-NODE-SVC] — 与 C++ ADEBUG 前缀 grep 一致。
# ═══════════════════════════════════════════════════════════════════════════════

import concurrent.futures as _cf


class AstroService:
    """
    Lightweight RPC service — replaces ``apollo::cyber::Service<Req,Resp>``.

    Registers a synchronous handler callable and dispatches incoming
    requests via ``call(request)`` → response.

    Topology registration mirrors Service::Init() →
    TopologyManager::service_manager()::Join(attr, ROLE_SERVER).
    ASTRO substitution: AstroChannelManager.join(service_name, ROLE="WRITER").
    """

    def __init__(
        self,
        node_name:    str,
        service_name: str,
        handler:      Callable[[Any], Any],
        host_name:    str = "localhost",
        process_id:   int = 0,
    ) -> None:
        self.service_name: str  = service_name
        self.node_name:    str  = node_name
        self._handler:     Callable[[Any], Any] = handler
        self._init:        bool = False

        # Topology: service server = WRITER role on the service channel
        self._ch_mgr = AstroChannelManager.instance()
        self._ch_mgr.join(
            channel_path=service_name,
            node_name=node_name,
            role_type="WRITER",
            host_name=host_name,
            process_id=process_id,
        )
        self._init = True
        _dbg("ASTRO-NODE-SVC",
             f"AstroService init ok svc={service_name} node={node_name}")

    def call(self, request: Any) -> Any:
        """
        Dispatch a request to the handler synchronously.

        Mirrors Service<Req,Resp>::ServiceCallback invocation inside
        Service::HandleRequest().  Returns the handler's response.
        """
        if not self._init:
            raise RuntimeError(f"AstroService '{self.service_name}' not initialised")
        _dbg("ASTRO-NODE-SVC",
             f"AstroService call svc={self.service_name}")
        return self._handler(request)

    def shutdown(self) -> None:
        """Leave topology — mirrors Service destructor / Shutdown()."""
        if self._init:
            self._ch_mgr.leave(self.service_name, self.node_name, "WRITER")
            self._init = False


class AstroClient:
    """
    Lightweight RPC client — replaces ``apollo::cyber::Client<Req,Resp>``.

    Holds a reference to an AstroService and issues synchronous calls
    via ``send_request(request, timeout_s)``.

    Topology registration mirrors Client::Init() →
    TopologyManager::service_manager()::Join(attr, ROLE_CLIENT).
    ASTRO substitution: AstroChannelManager.join(service_name, ROLE="READER").
    """

    def __init__(
        self,
        node_name:    str,
        service_name: str,
        host_name:    str = "localhost",
        process_id:   int = 0,
    ) -> None:
        self.service_name: str  = service_name
        self.node_name:    str  = node_name
        self._init:        bool = False
        self._service:     Optional[AstroService] = None

        # Topology: service client = READER role on the service channel
        self._ch_mgr = AstroChannelManager.instance()
        self._ch_mgr.join(
            channel_path=service_name,
            node_name=node_name,
            role_type="READER",
            host_name=host_name,
            process_id=process_id,
        )
        self._init = True
        _dbg("ASTRO-NODE-SVC",
             f"AstroClient init ok svc={service_name} node={node_name}")

    def bind_service(self, service: AstroService) -> None:
        """
        Bind a local AstroService — simulates RTPS service discovery.

        In Apollo, Client discovers Service via TopologyManager change callbacks
        and stores the RoleAttributes.  Here we accept a direct reference so
        unit tests can wire client → server without network.
        """
        self._service = service
        _dbg("ASTRO-NODE-SVC",
             f"AstroClient bind svc={self.service_name} node={self.node_name}")

    def send_request(
        self,
        request: Any,
        timeout_s: float = 5.0,
    ) -> Optional[Any]:
        """
        SendRequest(request, timeout) — mirrors Client<Req,Resp>::SendRequest.

        Apollo original: serialises request, sends via RTPS, waits on a
        condition variable for the response future.

        ASTRO path: if a service is bound locally, delegates directly to
        AstroService.call() (same-process fast path).  Returns None on
        timeout or unbound service.
        """
        if not self._init:
            _dbg("ASTRO-NODE-SVC",
                 f"AstroClient.send_request: not init svc={self.service_name}")
            return None

        if self._service is None:
            _dbg("ASTRO-NODE-SVC",
                 f"AstroClient.send_request: no service bound svc={self.service_name}")
            return None

        _dbg("ASTRO-NODE-SVC",
             f"AstroClient.send_request svc={self.service_name} timeout={timeout_s}s")
        try:
            future: _cf.Future = _cf.ThreadPoolExecutor(max_workers=1).submit(
                self._service.call, request
            )
            return future.result(timeout=timeout_s)
        except _cf.TimeoutError:
            _dbg("ASTRO-NODE-SVC",
                 f"AstroClient.send_request timeout svc={self.service_name}")
            return None
        except Exception as exc:  # noqa: BLE001
            _dbg("ASTRO-NODE-SVC",
                 f"AstroClient.send_request exc={exc}")
            return None

    def shutdown(self) -> None:
        """Leave topology — mirrors Client destructor / Shutdown()."""
        if self._init:
            self._ch_mgr.leave(self.service_name, self.node_name, "READER")
            self._init = False


class AstroNodeServiceImpl:
    """
    Node service factory — Python port of ``NodeServiceImpl``.

    Creates and lifecycle-manages AstroService / AstroClient pairs for one
    logical node, keeping the same factory semantics as the C++ Impl:

      • create_service(service_name, handler) → AstroService (init'd)
      • create_client(service_name)           → AstroClient  (init'd)

    All created objects are topology-registered with AstroChannelManager so
    the snapshot() shows the service graph alongside the channel graph.

    ASTRO delta from NodeServiceImpl::CreateService / CreateClient:
      service_id = md5(service_name)[:16] — mirrors GlobalData::RegisterService.
      service_list_ / client_list_ → strong refs (Python GC, no weak_ptr needed).
    """

    def __init__(
        self,
        node_name:    str,
        host_name:    str = "localhost",
        process_id:   int = 0,
    ) -> None:
        if not node_name:
            raise ValueError("AstroNodeServiceImpl requires a non-empty node_name")
        self.node_name:   str = node_name
        self._host_name:  str = host_name
        self._process_id: int = process_id

        self._services: List[AstroService] = []
        self._clients:  List[AstroClient]  = []

        _dbg("ASTRO-NODE-SVC",
             f"NodeServiceImpl ctor node={node_name}")

    # ── CreateService ─────────────────────────────────────────────────────────

    def create_service(
        self,
        service_name: str,
        handler:      Callable[[Any], Any],
    ) -> Optional[AstroService]:
        """
        CreateService<Req,Resp>(service_name, callback) — mirrors NodeServiceImpl
        template method.

        Constructs AstroService (which calls Init() internally), registers with
        AstroChannelManager as ROLE_SERVER (WRITER), appends to service_list_.
        Returns None if service_name is empty.
        """
        if not service_name:
            _dbg("ASTRO-NODE-SVC",
                 "create_service: empty service_name — abort")
            return None

        svc = AstroService(
            node_name=self.node_name,
            service_name=service_name,
            handler=handler,
            host_name=self._host_name,
            process_id=self._process_id,
        )
        self._services.append(svc)

        svc_id = self._register_service(service_name)
        _dbg("ASTRO-NODE-SVC",
             f"create_service ok node={self.node_name} svc={service_name} "
             f"svc_id={svc_id}")
        return svc

    # ── CreateClient ──────────────────────────────────────────────────────────

    def create_client(self, service_name: str) -> Optional[AstroClient]:
        """
        CreateClient<Req,Resp>(service_name) — mirrors NodeServiceImpl template.

        Constructs AstroClient (Init() called internally), registers with
        AstroChannelManager as ROLE_CLIENT (READER), appends to client_list_.
        Returns None if service_name is empty.
        """
        if not service_name:
            _dbg("ASTRO-NODE-SVC",
                 "create_client: empty service_name — abort")
            return None

        cli = AstroClient(
            node_name=self.node_name,
            service_name=service_name,
            host_name=self._host_name,
            process_id=self._process_id,
        )
        self._clients.append(cli)

        svc_id = self._register_service(service_name)
        _dbg("ASTRO-NODE-SVC",
             f"create_client ok node={self.node_name} svc={service_name} "
             f"svc_id={svc_id}")
        return cli

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _register_service(service_name: str) -> str:
        """GlobalData::RegisterService — deterministic md5 hex id."""
        import hashlib as _hl
        return _hl.md5(service_name.encode()).hexdigest()[:16]

    # ── shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Shutdown — mirrors NodeServiceImpl destructor.

        Calls shutdown() on all services and clients so topology is updated.
        """
        for svc in self._services:
            svc.shutdown()
        for cli in self._clients:
            cli.shutdown()
        self._services.clear()
        self._clients.clear()
        _dbg("ASTRO-NODE-SVC",
             f"NodeServiceImpl shutdown node={self.node_name}")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# AstroUnderlayMessage — ported from
#   upstream/apollo-cyber/transport/rtps/underlay_message.h
#
# 原典：Apollo RtpsTransmitter / RtpsReceiver 透过 FastCDR 序列化 UnderlayMessage，
# 再由 eprosima fastrtps Participant 在 DDS/RTPS 网络上传输。
# UnderlayMessage 是 IDL 自动生成的 POD 结构体，字段：
#   int32_t m_timestamp — 发送端 epoch 时间戳（秒，截断为 int32）
#   int32_t m_seq       — 每条消息的单调递增序号
#   std::string m_data  — 序列化后的消息正文（任意字节串）
#   std::string m_datatype — 消息类型名（对应 proto MessageDescriptor.full_name）
#
# 鲁迅曰：这消息的信封，薄薄的四个字段，却要走遍 DDS 的山山水水。
# 打开来，不过是一个时间戳、一个序号、一堆字节和一个名字——正如一切旅途
# 的终点，都只剩下问：你是谁，你从哪里来。
#
# 算法改动（20% 规则）：
#   1. FastCDR serialize/deserialize → struct.pack / unpack + base64（Python 无 FastCDR）。
#   2. getMaxCdrSerializedSize / getCdrSerializedSize → max_serialized_size() /
#      serialized_size() 返回 int；两函数逻辑完整保留（变长字段 +4 对齐）。
#   3. isKeyDefined() → 类方法 is_key_defined()（Apollo IDL 总返回 false）。
#   4. serializeKey() → serialize_key() 为空操作（Key 未定义，同 C++ 实现）。
#   5. m_timestamp 改为 float（monotonic_ns() 微秒精度）；int32 语义保留在
#      serialise 路径（截断为 int32）以保持线格式兼容。
# ═══════════════════════════════════════════════════════════════════════════════

import struct as _struct
import base64 as _base64


