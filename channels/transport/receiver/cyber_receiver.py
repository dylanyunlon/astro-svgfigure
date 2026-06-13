import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════════════

class _CyberReceiverBase(AstroEndpoint):
    """
    Common base for CyberIntraReceiver and CyberRtpsReceiver.

    Mirrors Receiver<M> from receiver.h:
      • Holds MessageListener callback.
      • Provides on_new_message() which calls the listener.
      • Subclasses implement enable() / disable().

    ASTRO delta: listener signature is (msg, msg_info) — role_attr injected
    via closure in the concrete receiver, matching C++ OnNewMessage 3-arg form.
    """

    def __init__(
        self,
        attr: AstroRoleAttributes,
        msg_listener: Callable[[Any, Any], None],
    ) -> None:
        super().__init__(attr)
        self._msg_listener = msg_listener

    def on_new_message(self, msg: Any, msg_info: Any) -> None:
        """
        OnNewMessage — mirrors Receiver<M>::OnNewMessage(msg, msg_info).
        Calls the registered MessageListener callback.
        """
        try:
            self._msg_listener(msg, msg_info)
        except Exception as exc:  # noqa: BLE001
            _dbg("ASTRO-RX-BASE",
                 f"listener exc ch={self.attr_.channel_name} exc={exc}")





class CyberIntraReceiver(_CyberReceiverBase):
    """
    Intra-process receiver — Python port of IntraReceiver<M>.

    Enable()  → AstroIntraDispatcher.add_listener(channel_id, role_id, cb).
                 Mirrors: dispatcher_->AddListener<M>(attr_, bind(&OnNewMessage,…))
    Disable() → AstroIntraDispatcher.remove_listener(channel_id, role_id).
                 Mirrors: dispatcher_->RemoveListener<M>(attr_)
    Enable(opposite_attr)  → add_listener_filtered (opposite_attr overload).
    Disable(opposite_attr) → remove_listener_filtered.

    Algorithm delta:
      IntraDispatcherPtr → AstroIntraDispatcher singleton (already in file).
      AddListener template<M> → add_listener(channel_id, role_id, cb): cb is typed
      via on_new_message closure (no C++ std::bind; Python lambda captures self).
    """

    def __init__(
        self,
        attr: AstroRoleAttributes,
        msg_listener: Callable[[Any, Any], None],
    ) -> None:
        super().__init__(attr, msg_listener)
        # mirrors: dispatcher_ = IntraDispatcher::Instance()
        self._dispatcher: AstroIntraDispatcher = AstroIntraDispatcher.instance()
        self._role_id: str = f"intra_rx::{attr.channel_name}::{attr.id}"

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Enable() / Enable(opposite_attr).
        Mirrors IntraReceiver::Enable() → dispatcher_->AddListener<M>(attr_, cb).
        Mirrors IntraReceiver::Enable(opposite_attr) → AddListener<M>(attr_, opposite_attr, cb).
        """
        if opposite_attr is None:
            if self.enabled_:
                return
            self._dispatcher.add_listener(
                self.attr_.channel_name,
                self._role_id,
                self.on_new_message,
            )
            self.enabled_ = True
            _dbg("ASTRO-INTRA-RX",
                 f"Enable ch={self.attr_.channel_name} role={self._role_id}")
        else:
            # opposite-attr filtered overload — no enabled_ guard (same as C++)
            oppo_id = f"{opposite_attr.channel_name}::{opposite_attr.id}"
            self._dispatcher.add_listener_filtered(
                self.attr_.channel_name,
                self._role_id,
                oppo_id,
                self.on_new_message,
            )
            _dbg("ASTRO-INTRA-RX",
                 f"Enable(filtered) ch={self.attr_.channel_name} oppo={oppo_id}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Disable() / Disable(opposite_attr).
        Mirrors IntraReceiver::Disable() → dispatcher_->RemoveListener<M>(attr_).
        Mirrors IntraReceiver::Disable(opposite_attr) → RemoveListener<M>(attr_, opposite_attr).
        """
        if opposite_attr is None:
            if not self.enabled_:
                return
            self._dispatcher.remove_listener(
                self.attr_.channel_name,
                self._role_id,
            )
            self.enabled_ = False
            _dbg("ASTRO-INTRA-RX",
                 f"Disable ch={self.attr_.channel_name}")
        else:
            oppo_id = f"{opposite_attr.channel_name}::{opposite_attr.id}"
            self._dispatcher.remove_listener_filtered(
                self.attr_.channel_name,
                self._role_id,
                oppo_id,
            )
            _dbg("ASTRO-INTRA-RX",
                 f"Disable(filtered) ch={self.attr_.channel_name} oppo={oppo_id}")





class CyberRtpsReceiver(_CyberReceiverBase):
    """
    RTPS receiver — Python port of RtpsReceiver<M>.

    Enable()  → AstroRtpsDispatcher.add_listener(attr, cb).
                 Mirrors: dispatcher_->AddListener<M>(attr_, bind(&OnNewMessage,…))
    Disable() → AstroRtpsDispatcher.remove_listener(attr).
                 Mirrors: dispatcher_->RemoveListener<M>(attr_)
    Enable(opposite_attr)  → filtered add (opposite_attr overload).
    Disable(opposite_attr) → filtered remove.

    Algorithm delta:
      RtpsDispatcherPtr → AstroRtpsDispatcher singleton (already in file).
      The raw-bytes payload from inject_message is passed directly to the
      listener; callers may deserialise via AstroUnderlayMessage.deserialize().
    """

    def __init__(
        self,
        attr: AstroRoleAttributes,
        msg_listener: Callable[[Any, Any], None],
    ) -> None:
        super().__init__(attr, msg_listener)
        # mirrors: dispatcher_ = RtpsDispatcher::Instance()
        self._dispatcher: AstroRtpsDispatcher = AstroRtpsDispatcher.instance()

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Enable() / Enable(opposite_attr).
        Mirrors RtpsReceiver::Enable() → dispatcher_->AddListener<M>(attr_, cb).
        Mirrors RtpsReceiver::Enable(opposite_attr) → AddListener<M>(attr_, opposite_attr, cb).
        """
        if opposite_attr is None:
            if self.enabled_:
                return
            self._dispatcher.add_listener(
                self.attr_,
                self.on_new_message,
            )
            self.enabled_ = True
            _dbg("ASTRO-RTPS-RX",
                 f"Enable ch={self.attr_.channel_name}")
        else:
            self._dispatcher.add_listener(
                self.attr_,
                self.on_new_message,
                opposite_attr,
            )
            _dbg("ASTRO-RTPS-RX",
                 f"Enable(filtered) ch={self.attr_.channel_name} "
                 f"oppo={opposite_attr.channel_name}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Disable() / Disable(opposite_attr).
        Mirrors RtpsReceiver::Disable() → dispatcher_->RemoveListener<M>(attr_).
        Mirrors RtpsReceiver::Disable(opposite_attr) → RemoveListener<M>(attr_, opposite_attr).
        """
        if opposite_attr is None:
            if not self.enabled_:
                return
            self._dispatcher.remove_listener(
                self.attr_,
                None,
            )
            self.enabled_ = False
            _dbg("ASTRO-RTPS-RX",
                 f"Disable ch={self.attr_.channel_name}")
        else:
            self._dispatcher.remove_listener(
                self.attr_,
                opposite_attr,
            )
            _dbg("ASTRO-RTPS-RX",
                 f"Disable(filtered) ch={self.attr_.channel_name} "
                 f"oppo={opposite_attr.channel_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# AstroComponentBase — ported from
#   upstream/apollo-cyber/component/component_base.h
#
# 原典：ComponentBase 是所有 cyber Component 的共同祖先，通过
# enable_shared_from_this 实现自引用，持有 node_（shared_ptr<Node>）、
# readers_（vector<shared_ptr<ReaderBase>>）、is_shutdown_（atomic<bool>）、
# config_file_path_（string）；
# Shutdown() 调用 Clear()、所有 reader 的 Shutdown()，再从 Scheduler
# 删除对应 Task（RemoveTask(node_->Name())）；
# GetProtoConfig<T> 从文件路径反序列化 protobuf 配置；
# LoadConfigFiles 将 ComponentConfig / TimerComponentConfig 中的
# config_file_path 和 flag_file_path 展开为绝对路径（APOLLO_CONF_PATH /
# APOLLO_FLAG_PATH 环境变量查找）。
#
# 鲁迅曰：基类的 Shutdown 先喊 Clear，再关 Reader，最后通知调度器——
# 这三步的顺序，不可颠倒，犹如善后的礼仪，次序即道德。
#
# 算法改动（20% 规则）：
#   1. enable_shared_from_this → 无（Python 引用计数自动管理）。
#   2. ComponentConfig / TimerComponentConfig proto → ComponentConf dataclass。
#   3. GetProtoConfig<T>: common::GetProtoFromFile → json.load (JSON 配置)。
#   4. LoadConfigFiles: GetFilePathWithEnv → _resolve_path（os.environ 查找）。
#   5. scheduler::Instance()->RemoveTask → AstroScheduler / LoopScheduler 均可；
#      此处记录 task_name 并在 shutdown() 内调用可选 scheduler.remove_task()。
#   6. readers_ vector<ReaderBase> → _readers: list[AstroCellReader]（鸭子类型）。
#   7. node_ shared_ptr<Node> → _node_name: str（Python 无 Node 类）。
#   8. Init() pure-virtual → 子类须实现 init() → bool。
#   9. Clear() 默认空实现 → clear() 默认空实现（子类可覆盖）。
#  10. flag_file_path 展开: google::SetCommandLineOption → os.environ 设置占位符
#      (标记 flagfile 路径；Python 无 gflags 依赖)。
#
# Debug prefix: [ASTRO-COMP] — 对应 C++ AINFO 前缀。
# ═══════════════════════════════════════════════════════════════════════════════

import abc as _abc_comp
import json as _json_comp
import os as _os_comp


@dataclasses.dataclass

