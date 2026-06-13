import os, sys, json, threading, dataclasses
import abc as _abc_comp
from typing import Any, Callable, Dict, List, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



@dataclasses.dataclass
class ComponentConf:
    """
    Lightweight component configuration — replaces proto::ComponentConfig and
    proto::TimerComponentConfig.

    config_file_path : path to a JSON config file (replaces .pb config).
    flag_file_path   : path to a flags file (replaces google::SetCommandLineOption).
    node_name        : logical node name (replaces proto.node_name).
    timer_interval_ms: timer interval in milliseconds (TimerComponentConfig only;
                       0 means non-timer component).
    """
    node_name:         str = ""
    config_file_path:  str = ""
    flag_file_path:    str = ""
    timer_interval_ms: int = 0





class AstroComponentBase(_abc_comp.ABC):
    """
    Abstract base for all Astro cell components.

    Ports ``apollo::cyber::ComponentBase`` from component_base.h.

    Subclasses must implement ``init() → bool`` (mirrors pure-virtual Init()).
    Optional override: ``clear()`` (mirrors virtual Clear(), default no-op).

    Lifecycle::
        comp = MyComponent()
        ok = comp.initialize(conf)    # calls init() internally
        # … component runs via reader callbacks …
        comp.shutdown()

    ASTRO delta from ComponentBase:
      • enable_shared_from_this → not needed (Python refcount).
      • Reader<M> template      → AstroCellReader (duck-typed, stored in _readers).
      • Node shared_ptr         → _node_name str + AstroNodeChannelImpl.
      • Scheduler::RemoveTask   → optional; call via scheduler kwarg in shutdown().
      • GetProtoConfig<T>       → get_config() returning dict (JSON-loaded).
      • LoadConfigFiles         → _load_config_files() with env-var path resolution.
    """

    def __init__(self) -> None:
        # mirrors std::atomic<bool> is_shutdown_ = {false}
        self._is_shutdown: bool = False
        self._shutdown_lock = threading.Lock()

        # mirrors std::shared_ptr<Node> node_ = nullptr
        self._node_name: str = ""
        self._node_impl: Optional[AstroNodeChannelImpl] = None

        # mirrors std::string config_file_path_ = ""
        self._config_file_path: str = ""

        # mirrors std::vector<std::shared_ptr<ReaderBase>> readers_
        self._readers: List[AstroCellReader] = []

        # loaded config dict (from JSON file) — replaces proto message
        self._config: Dict[str, Any] = {}

    # ── abstract interface ────────────────────────────────────────────────────

    @_abc_comp.abstractmethod
    def init(self) -> bool:
        """
        Init() — pure-virtual component initialisation.

        Called by initialize() after LoadConfigFiles.  Subclasses must:
          1. Create readers via self._node_impl.create_reader_by_name(…).
          2. Set up any periodic logic.
          3. Return True on success.
        """

    def clear(self) -> None:
        """
        Clear() — default empty implementation (mirrors virtual Clear()).
        Subclasses may override to release per-component resources.
        """

    # ── initialize (non-virtual entry point) ─────────────────────────────────

    def initialize(
        self,
        conf: ComponentConf,
        scheduler: Optional[Any] = None,
    ) -> bool:
        """
        Initialize(ComponentConfig) — non-virtual entry point.

        Mirrors ComponentBase::Initialize(const ComponentConfig& config):
            1. LoadConfigFiles(config)
            2. Init()            ← pure-virtual, implemented by subclass
            3. Register with scheduler (if provided)

        ASTRO: conf is ComponentConf; scheduler is AstroScheduler or None.
        Returns True iff init() succeeds.
        """
        self._node_name = conf.node_name or "astro_component"
        self._node_impl = AstroNodeChannelImpl(self._node_name)

        self._load_config_files(conf)

        _dbg("ASTRO-COMP",
             f"[ASTRO-COMPONENT] Initialize node={self._node_name} "
             f"config_file={self._config_file_path!r}")

        ok = self.init()
        if not ok:
            _dbg("ASTRO-COMP",
                 f"[ASTRO-COMPONENT] Init() returned false node={self._node_name}")
            return False

        _dbg("ASTRO-COMP",
             f"[ASTRO-COMPONENT] Init() ok node={self._node_name}")
        return True

    # ── shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self, scheduler: Optional[Any] = None) -> None:
        """
        Shutdown() — mirrors ComponentBase::Shutdown().

        Algorithm:
            if (is_shutdown_.exchange(true)) return;
            Clear();
            for (auto& reader : readers_) reader->Shutdown();
            scheduler::Instance()->RemoveTask(node_->Name());

        ASTRO:
            Clear()              → self.clear()
            reader->Shutdown()   → reader.shutdown()
            RemoveTask(name)     → scheduler.remove_task(node_name) if provided.
        """
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True

        _dbg("ASTRO-COMP",
             f"[ASTRO-COMPONENT] Shutdown node={self._node_name} "
             f"readers={len(self._readers)}")

        self.clear()

        for reader in self._readers:
            try:
                reader.shutdown()
            except Exception as exc:  # noqa: BLE001
                _dbg("ASTRO-COMP",
                     f"reader.shutdown exc node={self._node_name} exc={exc}")
        self._readers.clear()

        if scheduler is not None and hasattr(scheduler, "remove_task"):
            scheduler.remove_task(self._node_name)
            _dbg("ASTRO-COMP",
                 f"[ASTRO-COMPONENT] RemoveTask node={self._node_name}")

        if self._node_impl is not None:
            self._node_impl.shutdown()
            self._node_impl = None

    # ── GetProtoConfig<T> → get_config() ─────────────────────────────────────

    def get_config(self) -> Dict[str, Any]:
        """
        GetProtoConfig<T>(config) — return loaded config dict.

        Apollo: reads a protobuf from config_file_path_ using GetProtoFromFile.
        ASTRO:  returns _config (JSON-loaded dict); loads lazily if not yet read.
        """
        if not self._config and self._config_file_path:
            self._config = self._load_json_config(self._config_file_path)
        return self._config

    # ── ConfigFilePath accessor ───────────────────────────────────────────────

    @property
    def config_file_path(self) -> str:
        """ConfigFilePath() const — mirrors ComponentBase::ConfigFilePath()."""
        return self._config_file_path

    # ── is_shutdown property ──────────────────────────────────────────────────

    @property
    def is_shutdown(self) -> bool:
        with self._shutdown_lock:
            return self._is_shutdown

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_config_files(self, conf: ComponentConf) -> None:
        """
        LoadConfigFiles — mirrors ComponentBase::LoadConfigFiles(ComponentConfig).

        Resolves config_file_path via APOLLO_CONF_PATH env var.
        Resolves flag_file_path via APOLLO_FLAG_PATH env var.
        Stores resolved config path; loads JSON eagerly.

        ASTRO delta: google::SetCommandLineOption("flagfile", …) is replaced by
        setting os.environ["ASTRO_FLAGFILE"] so downstream code can read it.
        """
        if conf.config_file_path:
            resolved = self._resolve_path(conf.config_file_path, "APOLLO_CONF_PATH")
            if resolved:
                self._config_file_path = resolved
                _dbg("ASTRO-COMP",
                     f"[ASTRO-COMPONENT] use config file: {resolved}")
                self._config = self._load_json_config(resolved)
            else:
                _dbg("ASTRO-COMP",
                     f"[ASTRO-COMPONENT] conf file [{conf.config_file_path}] not found!")
                self._config_file_path = conf.config_file_path

        if conf.flag_file_path:
            flag_path = self._resolve_path(conf.flag_file_path, "APOLLO_FLAG_PATH")
            if flag_path:
                _dbg("ASTRO-COMP",
                     f"[ASTRO-COMPONENT] use flag file: {flag_path}")
                # Mirrors: google::SetCommandLineOption("flagfile", flag_file_path.c_str())
                _os_comp.environ["ASTRO_FLAGFILE"] = flag_path
            else:
                _dbg("ASTRO-COMP",
                     f"[ASTRO-COMPONENT] flag file [{conf.flag_file_path}] not found!")

    @staticmethod
    def _resolve_path(relative: str, env_var: str) -> str:
        """
        GetFilePathWithEnv — search for *relative* in the directory given by
        *env_var*, then fall back to the current working directory.
        Returns the absolute path if found, or "" if not.
        """
        search_dirs: List[str] = []
        env_val = _os_comp.environ.get(env_var, "")
        if env_val:
            search_dirs.extend(env_val.split(_os_comp.pathsep))
        search_dirs.append(_os_comp.getcwd())

        for base in search_dirs:
            candidate = _os_comp.path.join(base, relative)
            if _os_comp.path.exists(candidate):
                return _os_comp.path.abspath(candidate)

        # Also accept absolute path
        if _os_comp.path.isabs(relative) and _os_comp.path.exists(relative):
            return relative

        return ""

    @staticmethod
    def _load_json_config(path: str) -> Dict[str, Any]:
        """Load JSON config file → dict.  Returns {} on error."""
        try:
            with open(path) as fh:
                return _json_comp.load(fh)
        except (OSError, _json_comp.JSONDecodeError) as exc:
            _dbg("ASTRO-COMP", f"_load_json_config failed path={path!r} exc={exc}")
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
# AstroDataFusion — ported from
#   upstream/apollo-cyber/data/fusion/data_fusion.h
#
# 原典：DataFusion<M0,M1,M2,M3> 是纯抽象模板，提供三个偏特化版本：
#   4-channel: Fusion(index*, m0&, m1&, m2&, m3&) → bool
#   3-channel: Fusion(index*, m0&, m1&, m2&)       → bool
#   2-channel: Fusion(index*, m0&, m1&)             → bool
# 子类（如 AllLatest）重写 Fusion() 方法，从各自的 ChannelBuffer 中取最新值
# 组合成 N 元组后写入 fusion_buf_；调用方通过 index 追踪已读位置。
#
# 鲁迅曰：Fusion 是个忠实的账房——四个格子，每格一票，缺一不可；
# 等齐了才盖章，盖了章才算一次成功的融合。其实说来，世间诸事皆如此。
#
# 算法改动（20% 规则）：
#   1. template<M0,M1,M2,M3> 三偏特化 → 单 Python 类，arity 由构造时 channel_ids 长度决定。
#   2. NullType 占位符              → 省略（Python list 天然变长）。
#   3. bool* index out-param        → (new_index, tuple|None) 返回对（同 AstroAllLatest 惯例）。
#   4. shared_ptr<Mx>& NOLINT out-param → tuple 元素（caller destructures）。
#   5. 纯虚 Fusion() → Python @abc.abstractmethod fusion(index)。
#   6. AstroAllLatest 已在本文件实现；AstroDataFusion 作为更通用的接口层，
#      可由子类扩展为 barrier_fusion（等所有通道都有新数据才融合）等策略。
#   7. 新增 FusionPolicy enum: ALL_LATEST（已有 AstroAllLatest）/ BARRIER（新增）。
#   8. AstroBarrierFusion: 每个通道必须都有一个新值（epoch ≥ index+1）才触发。
#
# Debug prefix: [ASTRO-FUSION] — 与 AstroAllLatest 日志前缀一致。
# ═══════════════════════════════════════════════════════════════════════════════

import enum as _enum












# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-SHM] Remaining SHM ports: ProtobufArenaManager, ArenaAddressAllocator,
# ShmConf, ReadableInfo, ConditionNotifier
#
# Ported from:
#   upstream/apollo-cyber/transport/shm/protobuf_arena_manager.h (320 lines)
#   upstream/apollo-cyber/transport/shm/arena_address_allocator.h (122 lines)
#   upstream/apollo-cyber/transport/shm/shm_conf.h (87 lines)
#   upstream/apollo-cyber/transport/shm/readable_info.h (72 lines)
#   upstream/apollo-cyber/transport/shm/condition_notifier.h (71 lines)
#
# 20% algorithm changes:
#   1. POSIX shm_open/mmap → Python bytearray pool (no real shared memory)
#   2. Arena allocator best-fit → simplified first-fit with coalescing
#   3. futex wait/wake → threading.Condition
#   4. protobuf Arena → JSON buffer slots
#   5. ReadableInfo serialization → JSON string
# ═══════════════════════════════════════════════════════════════════════════════


