import os, sys, json, threading, abc as _abc, dataclasses, time as _time
from typing import Any, Callable, Dict, List, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



class AstroWarehouseBase(_abc.ABC):
    """Abstract warehouse interface — port of WarehouseBase."""

    @_abc.abstractmethod
    def add(self, key: str, role: "RoleRecord", ignore_if_exist: bool = True) -> bool: ...
    @_abc.abstractmethod
    def clear(self) -> None: ...
    @_abc.abstractmethod
    def size(self) -> int: ...
    @_abc.abstractmethod
    def remove_by_key(self, key: str) -> None: ...
    @_abc.abstractmethod
    def remove_by_role(self, key: str, role: "RoleRecord") -> None: ...
    @_abc.abstractmethod
    def remove_by_attr(self, attr: "RoleRecord") -> None: ...
    @_abc.abstractmethod
    def search_key(self, key: str) -> bool: ...
    @_abc.abstractmethod
    def search_first(self, key: str) -> Optional["RoleRecord"]: ...
    @_abc.abstractmethod
    def search_all(self, key: str) -> List["RoleRecord"]: ...
    @_abc.abstractmethod
    def search_attr(self, attr: "RoleRecord") -> bool: ...
    @_abc.abstractmethod
    def search_attr_first(self, attr: "RoleRecord") -> Optional["RoleRecord"]: ...
    @_abc.abstractmethod
    def search_attr_all(self, attr: "RoleRecord") -> List["RoleRecord"]: ...
    @_abc.abstractmethod
    def get_all_roles(self) -> List["RoleRecord"]: ...

    @staticmethod
    def _matches(role: "RoleRecord", attr: "RoleRecord") -> bool:
        if attr.channel_path and role.channel_path != attr.channel_path:
            return False
        if attr.node_name and role.node_name != attr.node_name:
            return False
        return True





class AstroSingleValueWarehouse(AstroWarehouseBase):
    """
    One role per key — mirrors SingleValueWarehouse (unordered_map).

    鲁迅曰：单值仓库是专制的——一把钥匙只开一扇门，
    新来的若想入住，先得把旧的赶走。
    """

    def __init__(self) -> None:
        self._roles: Dict[str, "RoleRecord"] = {}
        self._lock  = threading.RLock()

    def add(self, key: str, role: "RoleRecord", ignore_if_exist: bool = True) -> bool:
        with self._lock:
            if key in self._roles and ignore_if_exist:
                return False
            self._roles[key] = role
        return True

    def clear(self) -> None:
        with self._lock: self._roles.clear()

    def size(self) -> int:
        with self._lock: return len(self._roles)

    def remove_by_key(self, key: str) -> None:
        with self._lock: self._roles.pop(key, None)

    def remove_by_role(self, key: str, role: "RoleRecord") -> None:
        with self._lock:
            if self._roles.get(key) and self._roles[key].role_id == role.role_id:
                del self._roles[key]

    def remove_by_attr(self, attr: "RoleRecord") -> None:
        with self._lock:
            for k in [k for k, r in self._roles.items() if self._matches(r, attr)]:
                del self._roles[k]

    def search_key(self, key: str) -> bool:
        with self._lock: return key in self._roles

    def search_first(self, key: str) -> Optional["RoleRecord"]:
        with self._lock: return self._roles.get(key)

    def search_all(self, key: str) -> List["RoleRecord"]:
        with self._lock:
            r = self._roles.get(key); return [r] if r else []

    def search_attr(self, attr: "RoleRecord") -> bool:
        with self._lock: return any(self._matches(r, attr) for r in self._roles.values())

    def search_attr_first(self, attr: "RoleRecord") -> Optional["RoleRecord"]:
        with self._lock:
            for r in self._roles.values():
                if self._matches(r, attr): return r
            return None

    def search_attr_all(self, attr: "RoleRecord") -> List["RoleRecord"]:
        with self._lock: return [r for r in self._roles.values() if self._matches(r, attr)]

    def get_all_roles(self) -> List["RoleRecord"]:
        with self._lock: return list(self._roles.values())





class AstroMultiValueWarehouse(AstroWarehouseBase):
    """
    Multiple roles per key — mirrors MultiValueWarehouse (unordered_multimap).

    鲁迅曰：多值仓库是民主的——一把钥匙后面站着一排人，
    新来者也可以挤进去，只要还没有人和你同名同姓。
    """

    def __init__(self) -> None:
        self._roles: Dict[str, List["RoleRecord"]] = {}
        self._lock  = threading.RLock()

    def add(self, key: str, role: "RoleRecord", ignore_if_exist: bool = True) -> bool:
        with self._lock:
            bucket = self._roles.setdefault(key, [])
            if ignore_if_exist and any(r.role_id == role.role_id for r in bucket):
                return False
            bucket.append(role)
        return True

    def clear(self) -> None:
        with self._lock: self._roles.clear()

    def size(self) -> int:
        with self._lock: return sum(len(v) for v in self._roles.values())

    def remove_by_key(self, key: str) -> None:
        with self._lock: self._roles.pop(key, None)

    def remove_by_role(self, key: str, role: "RoleRecord") -> None:
        with self._lock:
            if key in self._roles:
                self._roles[key] = [r for r in self._roles[key] if r.role_id != role.role_id]
                if not self._roles[key]: del self._roles[key]

    def remove_by_attr(self, attr: "RoleRecord") -> None:
        with self._lock:
            for key in list(self._roles):
                self._roles[key] = [r for r in self._roles[key] if not self._matches(r, attr)]
                if not self._roles[key]: del self._roles[key]

    def search_key(self, key: str) -> bool:
        with self._lock: return bool(self._roles.get(key))

    def search_first(self, key: str) -> Optional["RoleRecord"]:
        with self._lock:
            b = self._roles.get(key); return b[0] if b else None

    def search_all(self, key: str) -> List["RoleRecord"]:
        with self._lock: return list(self._roles.get(key, []))

    def search_attr(self, attr: "RoleRecord") -> bool:
        with self._lock:
            return any(self._matches(r, attr) for b in self._roles.values() for r in b)

    def search_attr_first(self, attr: "RoleRecord") -> Optional["RoleRecord"]:
        with self._lock:
            for b in self._roles.values():
                for r in b:
                    if self._matches(r, attr): return r
            return None

    def search_attr_all(self, attr: "RoleRecord") -> List["RoleRecord"]:
        with self._lock:
            return [r for b in self._roles.values() for r in b if self._matches(r, attr)]

    def get_all_roles(self) -> List["RoleRecord"]:
        with self._lock:
            return [r for b in self._roles.values() for r in b]


# ═══════════════════════════════════════════════════════════════════════════════
# AstroChoreographyScheduler — ported from
#   upstream/apollo-cyber/scheduler/policy/scheduler_choreography.h
#
# 原典：SchedulerChoreography 继承 Scheduler，引入"编舞"概念：
#   choreography processors — 绑定 cr_confs_ 中指定任务的专用核。
#   pool processors         — 处理其余任务的通用线程池。
#
# 鲁迅曰：编舞者把舞台分成两半——台前的明星各自占据指定的 CPU，
# 台后的杂役共用一个线程池；分工清晰，却都在同一幕戏里表演。
#
# 算法改动（20% 规则）：
#   1. ChoreographyTask proto   → ChoreographyTaskConf dataclass（name/priority/processor_id）。
#   2. CRoutine (coroutine)     → Python callable。
#   3. CPU affinity / OS policy → 仅记录，不实际调用 pthread_setaffinity。
#   4. choreography processors  → _ChoreographyProcessor（私有 heapq/slot）。
#   5. pool processors          → _PoolProcessor（共享 pool heapq）。
#   6. DispatchTask             → 按 cr_confs_ 路由到 chore / pool。
#   7. NotifyProcessor          → chore 任务唤醒专用 processor；pool 任务唤醒任意池 proc。
#   8. CreateProcessor          → __init__ 中按 conf 分配两类 Processor。
#   9. RemoveTask/RemoveCRoutine→ 从 cr_confs_、chore_map、task registry 三处清除。
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# IntraTransmitter + RtpsTransmitter + IntraReceiver + RtpsReceiver
# Ported from:
#   upstream/apollo-cyber/transport/transmitter/intra_transmitter.h
#   upstream/apollo-cyber/transport/transmitter/rtps_transmitter.h
#   upstream/apollo-cyber/transport/receiver/intra_receiver.h
#   upstream/apollo-cyber/transport/receiver/rtps_receiver.h
#
# 鲁迅曰：进程内的消息走不出墙，RTPS 的消息走不进家——两套发送者，
# 两套接收者，却都在同一个调度循环里假装彼此无关。共享分发器是这出
# 戏的真正主角，transmitter 与 receiver 不过是它两侧的走卒。
#
# 算法改动（20% 规则）：
#   1. template<M>               → duck-typed Python（msg: Any）。
#   2. IntraDispatcher singleton → AstroIntraDispatcher.instance()（已有）。
#   3. RtpsDispatcher singleton  → AstroRtpsDispatcher.instance()（已有）。
#   4. enabled_ bool             → _enabled bool（命名对齐 Python 惯例）。
#   5. AcquireMessage() → 返回 {} （无 arena；同本文件其他 transmitter）。
#   6. RtpsTransmitter::Enable 创建 fastrtps Publisher
#      → 仅记录 endpoint；Transmit 调用 AstroRtpsDispatcher.inject_message()。
#   7. RtpsTransmitter::Transmit UnderlayMessage + WriteParams
#      → AstroUnderlayMessage.serialize() + AstroMessageInfo 两路打包。
#   8. IntraReceiver  AddListener/RemoveListener → AstroIntraDispatcher 对应方法。
#   9. RtpsReceiver   AddListener/RemoveListener → AstroRtpsDispatcher 对应方法。
#  10. OnNewMessage callback signature: (msg, msg_info, role_attr) → (msg, msg_info)
#      （role_attr 通过闭包注入，保持与 C++ 三参形式等价）。
#
# Debug prefix: [ASTRO-INTRA-TX] / [ASTRO-RTPS-TX] /
#               [ASTRO-INTRA-RX] / [ASTRO-RTPS-RX]
# ═══════════════════════════════════════════════════════════════════════════════


