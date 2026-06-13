import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)


# --- transport/qos/qos_profile_conf.h (61 lines) ---
class AstroQoSProfileConf:
    """Port of QoSProfileConf — Quality of Service configuration."""
    def __init__(self):
        self.history_depth = 10
        self.reliability = "RELIABLE"  # or "BEST_EFFORT"
        self.durability = "VOLATILE"   # or "TRANSIENT_LOCAL"
        self.mps = 0  # messages per second limit (0 = unlimited)
    @classmethod
    def default(cls): return cls()
    def to_dict(self) -> dict:
        return {"history_depth": self.history_depth, "reliability": self.reliability,
                "durability": self.durability, "mps": self.mps}


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CC] Apollo .cc implementations — core algorithm ports
#
# These supplement the class skeletons from .h ports with actual logic
# from the corresponding .cc files. 49 .cc files → batch ported.
#
# 20% changes: C++ → Python idioms, POSIX → threading, protobuf → dict
# ═══════════════════════════════════════════════════════════════════════════════

# --- graph.cc (284 lines): edge insert/delete/BFS traversal ---
def _graph_insert_edge(graph_dict: dict, src: str, dst: str, edge_id: str = ""):
    """Port of Graph::Insert — bidirectional edge insertion."""
    graph_dict.setdefault(src, {"out": [], "in": []})
    graph_dict.setdefault(dst, {"out": [], "in": []})
    edge = {"src": src, "dst": dst, "id": edge_id or f"{src}->{dst}"}
    graph_dict[src]["out"].append(edge)
    graph_dict[dst]["in"].append(edge)
    return edge

def _graph_delete_edge(graph_dict: dict, src: str, dst: str):
    """Port of Graph::Delete."""
    if src in graph_dict:
        graph_dict[src]["out"] = [e for e in graph_dict[src]["out"] if e["dst"] != dst]
    if dst in graph_dict:
        graph_dict[dst]["in"] = [e for e in graph_dict[dst]["in"] if e["src"] != src]

def _graph_bfs(graph_dict: dict, start: str, end: str) -> bool:
    """Port of Graph::LevelTraverse — BFS reachability check."""
    from collections import deque
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            return True
        if node in visited:
            continue
        visited.add(node)
        for edge in graph_dict.get(node, {}).get("out", []):
            queue.append(edge["dst"])
    return False

def _graph_topo_sort(graph_dict: dict) -> list:
    """Topological sort (Kahn's algorithm) — used for epoch execution order."""
    in_degree = {n: len(graph_dict[n]["in"]) for n in graph_dict}
    from collections import deque
    queue = deque([n for n, d in in_degree.items() if d == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for edge in graph_dict.get(node, {}).get("out", []):
            dst = edge["dst"]
            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                queue.append(dst)
    return order


# --- scheduler.cc (196 lines): task creation + notification ---
def _scheduler_create_task(scheduler, func, name: str, priority: int = 0) -> int:
    """Port of Scheduler::CreateTask — register a cell constraint coroutine."""
    import hashlib
    task_id = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    scheduler._tasks[task_id] = {
        "name": name, "func": func, "priority": priority,
        "state": "ready", "epoch_count": 0
    }
    _dbg("ASTRO-SCHEDULER", f"CreateTask name={name} taskId={task_id}")
    return task_id

def _scheduler_notify_task(scheduler, task_id: int) -> bool:
    """Port of Scheduler::NotifyTask — trigger epoch tick for a task."""
    task = scheduler._tasks.get(task_id)
    if not task:
        return False
    task["state"] = "notified"
    _dbg("ASTRO-SCHEDULER", f"NotifyTask taskId={task_id} name={task['name']}")
    return True

def _scheduler_dispatch_all(scheduler):
    """Port of Scheduler main loop — execute all ready/notified tasks by priority."""
    sorted_tasks = sorted(scheduler._tasks.values(), key=lambda t: -t["priority"])
    for task in sorted_tasks:
        if task["state"] in ("ready", "notified"):
            task["state"] = "running"
            try:
                task["func"]()
                task["epoch_count"] += 1
            except Exception as e:
                _dbg("ASTRO-SCHEDULER", f"Task {task['name']} failed: {e}")
            task["state"] = "ready"


# --- topology_manager.cc (246 lines): init + change notification ---
def _topology_init(topo_mgr):
    """Port of TopologyManager::Init — initialize all sub-managers."""
    topo_mgr._node_mgr = {}
    topo_mgr._channel_mgr = AstroChannelManager() if 'AstroChannelManager' in dir() else {}
    topo_mgr._change_listeners = []
    topo_mgr._initialized = True
    _dbg("ASTRO-TOPO", "TopologyManager initialized")
    return True

def _topology_on_change(topo_mgr, change_type: str, channel: str, role: str):
    """Port of TopologyManager::OnParticipantChange — fire change listeners."""
    event = {"type": change_type, "channel": channel, "role": role}
    for listener in topo_mgr._change_listeners:
        try:
            listener(event)
        except Exception as e:
            _dbg("ASTRO-TOPO", f"Change listener error: {e}")


# --- transport.cc (99 lines): factory initialization ---
def _transport_init(transport):
    """Port of Transport constructor — create dispatchers."""
    transport._intra_dispatcher = AstroIntraDispatcher() if 'AstroIntraDispatcher' in dir() else None
    transport._shutdown = False
    _dbg("ASTRO-TRANSPORT", "Transport initialized")

def _transport_shutdown(transport):
    """Port of Transport::Shutdown."""
    transport._shutdown = True
    if transport._intra_dispatcher:
        transport._intra_dispatcher.shutdown()
    _dbg("ASTRO-TRANSPORT", "Transport shutdown")


# --- channel_manager.cc (353 lines): join/leave/query ---
def _channel_mgr_join(mgr, channel_name: str, role_type: str, node_id: str = ""):
    """Port of ChannelManager::DisposeJoin."""
    mgr._channels.setdefault(channel_name, {"writers": [], "readers": []})
    if role_type == "writer":
        mgr._channels[channel_name]["writers"].append(node_id)
    else:
        mgr._channels[channel_name]["readers"].append(node_id)
    _dbg("ASTRO-CHANNEL-MGR", f"Join ch={channel_name} role={role_type} node={node_id}")
    # Fire topo change
    for cb in mgr._change_callbacks:
        cb({"type": "join", "channel": channel_name, "role": role_type})

def _channel_mgr_leave(mgr, channel_name: str, role_type: str, node_id: str = ""):
    """Port of ChannelManager::DisposeLeave."""
    ch = mgr._channels.get(channel_name, {})
    key = "writers" if role_type == "writer" else "readers"
    if node_id in ch.get(key, []):
        ch[key].remove(node_id)
    for cb in mgr._change_callbacks:
        cb({"type": "leave", "channel": channel_name, "role": role_type})


# --- warehouse.cc (single_value 208 + multi_value 217 lines) ---
def _warehouse_add(warehouse: dict, key: str, value):
    """Port of SingleValueWarehouse::Add / MultiValueWarehouse::Add."""
    warehouse[key] = value

def _warehouse_search(warehouse: dict, key: str):
    """Port of Warehouse::Search."""
    return warehouse.get(key)

def _warehouse_get_all_keys(warehouse: dict) -> list:
    """Port of Warehouse::GetAllRoles → keys."""
    return list(warehouse.keys())


# --- condition_notifier.cc (210 lines): wait/notify with timeout ---
def _condition_notify_with_info(notifier, channel_id: str, block_idx: int = 0):
    """Port of ConditionNotifier::Notify with ReadableInfo."""
    info = AstroReadableInfo(host_id=0, block_index=block_idx, channel_id=channel_id)
    return notifier.notify(info)


# --- shm_dispatcher.cc (232 lines): shared memory message dispatch ---
def _shm_dispatch_message(dispatcher, channel_id: str, msg: dict):
    """Port of ShmDispatcher::OnMessage — dispatch from shared memory."""
    listeners = dispatcher._listeners.get(channel_id, [])
    info = AstroMessageInfo() if 'AstroMessageInfo' in dir() else {}
    for listener_cb in listeners:
        try:
            listener_cb(msg, info)
        except Exception as e:
            _dbg("ASTRO-SHM-DISPATCH", f"Listener error on ch={channel_id}: {e}")


# --- scheduler_classic.cc (225 lines) + classic_context.cc (124 lines) ---
def _classic_context_wait(ctx, timeout: float = 0.1):
    """Port of ClassicContext::Wait — wait for task notification."""
    ctx._notified.wait(timeout)
    ctx._notified.clear()

def _classic_schedule_loop(scheduler):
    """Port of SchedulerClassic main loop — round-robin all processors."""
    import time
    while not scheduler._shutdown:
        _scheduler_dispatch_all(scheduler)
        time.sleep(0.001)  # 1ms tick


# --- message_info.cc (141 lines): serialization ---
def _message_info_serialize(info) -> dict:
    """Port of MessageInfo serialization."""
    return {
        "sender_id": str(getattr(info, 'sender_id', '')),
        "channel_id": str(getattr(info, 'channel_id', '')),
        "seq_num": getattr(info, 'seq_num', 0),
        "send_time": getattr(info, 'send_time', 0.0),
    }


# --- role.cc (86 lines): role matching ---
def _role_match(role_a: dict, role_b: dict) -> bool:
    """Port of RoleAttributes matching logic."""
    return (role_a.get("channel_name") == role_b.get("channel_name") and
            role_a.get("message_type") == role_b.get("message_type"))


# --- participant.cc (145 lines): network participant ---
def _participant_create_publisher(participant, channel: str, qos: dict = None) -> dict:
    """Port of Participant::CreatePublisher."""
    attrs = AstroAttributesFiller.fill_publisher_attrs(channel, qos)
    _dbg("ASTRO-RTPS", f"CreatePublisher ch={channel}")
    return attrs

def _participant_create_subscriber(participant, channel: str, listener_cb=None, qos: dict = None) -> dict:
    """Port of Participant::CreateSubscriber."""
    attrs = AstroAttributesFiller.fill_subscriber_attrs(channel, qos)
    if listener_cb:
        sub_listener = AstroSubListener(callback=listener_cb)
        attrs["_listener"] = sub_listener
    _dbg("ASTRO-RTPS", f"CreateSubscriber ch={channel}")
    return attrs


# --- scheduler_choreography.cc (266 lines): DAG-based execution ---
def _choreography_dispatch(scheduler, graph: dict):
    """Port of SchedulerChoreography — execute tasks in topological order."""
    order = _graph_topo_sort(graph)
    for node_id in order:
        task = scheduler._tasks.get(node_id)
        if task and task["state"] in ("ready", "notified"):
            task["state"] = "running"
            try:
                task["func"]()
                task["epoch_count"] += 1
            except Exception as e:
                _dbg("ASTRO-CHOREO", f"Task {node_id} failed: {e}")
            task["state"] = "ready"

