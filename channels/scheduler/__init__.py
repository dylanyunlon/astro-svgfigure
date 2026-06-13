"""
channels/scheduler — scheduler classes split from channel_runtime.py

Includes: LoopScheduler, AstroSnapshot, AstroProcessor, AstroScheduler,
          ChoreographyTaskConf, _ChoreographyProcessor, _PoolProcessor,
          AstroChoreographyScheduler, AstroSchedulerFactory,
          AstroChoreographyContext, AstroCvWrapper, AstroMutexWrapper,
          AstroPinThread, AstroProcessorContext
"""

import heapq
import concurrent.futures
import dataclasses
import threading
import threading as _cv_threading
import time
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

# Re-use CHANNELS_DIR and _dbg from the parent package at runtime;
# import them lazily to avoid a circular import at module-load time.
def _get_channels_dir():
    import os as _os
    return _os.path.dirname(_os.path.abspath(__file__ + "/.."))

# _dbg is defined in channel_runtime; import lazily.
def _dbg(prefix: str, msg: str):
    try:
        from channels.channel_runtime import _dbg as _real_dbg
        _real_dbg(prefix, msg)
    except Exception:
        pass

CHANNELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class LoopScheduler:
    """
    Cell proc() scheduler ordered by z-layer priority.

    Mirrors apollo::cyber::scheduler::Scheduler with:
      task_name  → cell_id  (string)
      priority   → z-layer  (int, lower = higher scheduling priority)
      processor  → proc_fn  callable

    Usage:
        sched = LoopScheduler()
        sched.register_cell("input_embed", z=3)
        sched.register_cell("self_attn",   z=3)
        sched.run_epoch(proc)        # calls proc("input_embed"), proc("self_attn") …
    """

    def __init__(self):
        # List of (z, insertion_seq, cell_id) — kept unsorted until run_epoch.
        self._tasks: List[Tuple[int, int, str]] = []
        self._seq = 0
        # z_layers channel path (relative to CHANNELS_DIR)
        self._z_layers_path = "physics/z_layers.json"
        _dbg("ASTRO-SCHED", "LoopScheduler constructed")

    def register_cell(self, cell_id: str, z: Optional[int] = None):
        """
        RegisterCell — add a cell to the scheduler.

        If z is None the scheduler reads physics/z_layers.json to resolve
        the z value; falls back to z=3 (default node layer) if not found.
        Mirrors Scheduler::CreateTask with priority auto-detection.
        """
        if z is None:
            z = self._resolve_z(cell_id)
        self._tasks.append((z, self._seq, cell_id))
        self._seq += 1
        _dbg("ASTRO-SCHED", f"register cell={cell_id} z={z} seq={self._seq - 1}")

    def register_cells_from_z_layers(self):
        """
        Bulk-register all cells present in physics/z_layers.json.
        Convenience wrapper — mirrors Scheduler::CreateTaskBatch.
        """
        full = os.path.join(CHANNELS_DIR, self._z_layers_path)
        if not os.path.exists(full):
            _dbg("ASTRO-SCHED", "register_cells_from_z_layers: z_layers.json missing, skip")
            return
        with open(full) as f:
            z_data = json.load(f)
        for cell_id, val in z_data.items():
            if cell_id == "__schema__":
                continue
            z = val if isinstance(val, int) else val.get("z", 3)
            self.register_cell(cell_id, z=z)

    def run_epoch(self, proc_fn: Callable[[str], Any]) -> List[str]:
        """
        run_epoch — dispatch all registered cells in z-layer priority order.

        Returns ordered list of dispatched cell_ids (useful for tests).
        Mirrors Scheduler::NotifyProcessor → ProcBalance dispatch loop.
        """
        ordered = sorted(self._tasks, key=lambda t: (t[0], t[1]))
        dispatched = []
        for z, seq, cell_id in ordered:
            _dbg("ASTRO-SCHED", f"dispatch cell={cell_id} z={z} seq={seq}")
            proc_fn(cell_id)
            dispatched.append(cell_id)
        return dispatched

    def sorted_cells(self) -> List[str]:
        """Return cells in dispatch order without executing. Useful for dry-run."""
        return [cell_id for _, _, cell_id in sorted(self._tasks, key=lambda t: (t[0], t[1]))]

    def clear(self):
        """Reset task list — mirrors Scheduler::Shutdown task drain."""
        self._tasks.clear()
        self._seq = 0

    def _resolve_z(self, cell_id: str) -> int:
        """Read z for cell_id from z_layers.json. Fallback = 3."""
        full = os.path.join(CHANNELS_DIR, self._z_layers_path)
        if not os.path.exists(full):
            return 3
        try:
            with open(full) as f:
                data = json.load(f)
            val = data.get(cell_id, 3)
            return val if isinstance(val, int) else val.get("z", 3)
        except (json.JSONDecodeError, OSError):
            return 3


@dataclasses.dataclass
class AstroSnapshot:
    """Per-processor execution snapshot — mirrors Snapshot struct."""
    processor_id: str = ""
    routine_name: str = ""
    execute_start_mono: float = 0.0   # 0.0 → idle (mirrors execute_start_time=0)


class AstroProcessor:
    """
    Executor thread for the classic-policy scheduler.

    Ports apollo::cyber::scheduler::Processor (processor.h / processor.cc).

    ASTRO deltas:
      • std::thread → ThreadPoolExecutor future (no explicit thread management).
      • CRoutine::Resume() → plain callable task().
      • context_->Wait() (condition_variable) → threading.Event.wait(timeout=0.01).
      • Snapshot fields preserved; execute_start_mono replaces execute_start_time.
    """

    def __init__(self, processor_id: str, scheduler: "AstroScheduler"):
        self._id: str = processor_id
        self._scheduler: "AstroScheduler" = scheduler
        self._running: bool = False
        self._future: Optional[concurrent.futures.Future] = None
        self._snap = AstroSnapshot(processor_id=processor_id)
        self._wake = threading.Event()
        _dbg("ASTRO-PROC", f"AstroProcessor ctor id={processor_id}")

    def bind_and_start(self, executor: concurrent.futures.ThreadPoolExecutor) -> None:
        """
        BindContext + thread start — mirrors Processor::BindContext(ctx) which
        calls std::call_once(thread_flag_, [this]{ thread_ = std::thread(Run, this) }).
        Submits _run() to the provided ThreadPoolExecutor.
        """
        if self._running:
            return
        self._running = True
        _dbg("ASTRO-PROC", f"AstroProcessor bind_and_start id={self._id} ONLINE")
        self._future = executor.submit(self._run)

    def stop(self) -> None:
        """Stop() — mirrors Processor::Stop(). Clears flag, wakes loop, joins."""
        if not self._running:
            return
        self._running = False
        self._wake.set()
        if self._future is not None:
            try:
                self._future.result(timeout=2.0)
            except (concurrent.futures.TimeoutError, Exception):
                pass
        _dbg("ASTRO-PROC", f"AstroProcessor stop id={self._id}")

    @property
    def snapshot(self) -> AstroSnapshot:
        """ProcSnapshot() — return live snapshot reference."""
        return self._snap

    def _run(self) -> None:
        """
        Processor::Run() spin loop — ported to Python.

        Each iteration:
            1. Dequeue the highest-priority ready task from the scheduler.
            2. If found: stamp snapshot, execute task(), clear snapshot.
            3. If none:  wait on threading.Event (mirrors context_->Wait()).
        """
        _dbg("ASTRO-PROC",
             f"[ASTRO-PROCESSOR] Processor::Run epoch executor id={self._id} ONLINE")
        while self._running:
            task_name, task_func = self._scheduler._dequeue_task()
            if task_func is not None:
                self._snap.execute_start_mono = time.monotonic()
                self._snap.routine_name = task_name
                _dbg("ASTRO-PROC",
                     f"[ASTRO-PROCESSOR] epoch tick: cell='{task_name}' proc={self._id}")
                try:
                    task_func()
                except Exception as exc:
                    _dbg("ASTRO-PROC",
                         f"task exc cell={task_name} proc={self._id} exc={exc}")
                finally:
                    self._snap.execute_start_mono = 0.0
                    self._snap.routine_name = ""
            else:
                self._wake.clear()
                self._wake.wait(timeout=0.01)

    def notify(self) -> None:
        """Wake the spin loop — mirrors ClassicContext::Notify → cv_wq_.notify_one()."""
        self._wake.set()


class AstroScheduler:
    """
    Classic-policy task scheduler with ThreadPoolExecutor processor pool.

    Ports apollo::cyber::scheduler::Scheduler + SchedulerClassic (scheduler.h,
    scheduler.cc, policy/scheduler_classic.h, policy/classic_context.h).

    Key design decisions:
      • task registry  : dict[task_name, callable]  (mirrors id_cr_ map)
      • dispatch queue : heapq[(z, seq, task_name)] (mirrors MULTI_PRIO_QUEUE)
      • processors     : list[AstroProcessor]       (mirrors processors_ vector)
      • executor       : ThreadPoolExecutor          (backs AstroProcessor threads)
      • epoch_index    : int                        (mirrors atomic<uint64_t> M126)

    Usage::

        sched = AstroScheduler(num_processors=2)
        sched.create_task(lambda: render("self_attn"), "self_attn", z=3)
        sched.create_task(lambda: render("input_embed"), "input_embed", z=1)
        sched.run_until_done()
        sched.shutdown()
    """

    def __init__(self, num_processors: int = 2):
        self._stop: bool = False
        self._epoch_index: int = 0

        self._tasks: Dict[str, Any] = {}
        self._task_lock = threading.Lock()

        self._queue: List[tuple] = []
        self._queue_lock = threading.Lock()
        self._queue_seq: int = 0

        self._num_processors: int = max(1, num_processors)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._num_processors,
            thread_name_prefix="astro_proc",
        )
        self._processors: List[AstroProcessor] = []
        self._pending: int = 0
        self._pending_lock = threading.Lock()
        self._all_done = threading.Event()
        self._all_done.set()

        for i in range(self._num_processors):
            proc = AstroProcessor(processor_id=f"proc_{i}", scheduler=self)
            proc.bind_and_start(self._executor)
            self._processors.append(proc)

        _dbg("ASTRO-SCHED",
             f"AstroScheduler constructed num_processors={self._num_processors}")

    # ── epoch control (M126 port) ──────────────────────────────────────────────

    def advance_epoch(self) -> int:
        """
        AdvanceEpoch() — mirrors Scheduler::AdvanceEpoch() (M126).
        Increments epoch_index_; returns the new epoch value.
        Called by the cell-pubsub loop at each SVG layout iteration.
        """
        self._epoch_index += 1
        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHED] epoch_advance epoch={self._epoch_index}")
        return self._epoch_index

    @property
    def current_epoch(self) -> int:
        """CurrentEpoch() — read-only snapshot of epoch_index_."""
        return self._epoch_index

    # ── CreateTask ────────────────────────────────────────────────────────────

    def create_task(
        self,
        func: Callable[[], None],
        task_name: str,
        z: int = 3,
        channel_path: Optional[str] = None,
    ) -> bool:
        """
        CreateTask(func, name, z, channel_path) — mirrors Scheduler::CreateTask.

        Registers the callable under task_name in the task registry, then
        calls _dispatch_task() to enqueue it into the priority heap.

        If channel_path is provided, registers a DataNotifier callback so that
        when data arrives on the channel the task is re-enqueued — mirroring
        visitor->RegisterNotifyCallback([this, task_id]{ NotifyProcessor(task_id) }).
        """
        if self._stop:
            _dbg("ASTRO-SCHED",
                 f"CreateTask: scheduler stopped, cannot create task name={task_name}")
            return False

        with self._task_lock:
            self._tasks[task_name] = func

        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHEDULER] CreateTask name={task_name} z={z} "
             f"hasChannel={channel_path is not None}")

        self._dispatch_task(task_name, z)

        if channel_path is not None:
            def _notify_cb():
                if not self._stop:
                    self.notify_processor(task_name, z)
            DataNotifier.instance().add_notifier(
                channel_path, Notifier(_notify_cb)
            )

        return True

    # ── NotifyProcessor ───────────────────────────────────────────────────────

    def notify_processor(self, task_name: str, z: int = 3) -> bool:
        """
        NotifyProcessor(task_name) — mirrors SchedulerClassic::NotifyProcessor.

        Re-enqueues the named task into the dispatch heap and wakes a processor.
        Called on incoming data notifications (channel callbacks).
        Returns False if the scheduler is stopped or task unknown.
        """
        if self._stop:
            return False

        with self._task_lock:
            if task_name not in self._tasks:
                _dbg("ASTRO-SCHED",
                     f"[ASTRO-SCHEDULER] NotifyProcessor: unknown task={task_name}")
                return False

        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHEDULER] NotifyProcessor task={task_name} z={z}")
        self._dispatch_task(task_name, z)
        return True

    # ── internal dispatch + dequeue ───────────────────────────────────────────

    def _dispatch_task(self, task_name: str, z: int) -> None:
        """
        _dispatch_task — DispatchTask analogue.

        Pushes (z, seq, task_name) onto the heapq so the lowest-z task is
        served first, mirroring ClassicContext's MULTI_PRIO_QUEUE ordering.
        Then wakes one processor (ClassicContext::Notify → cv_wq_.notify_one()).
        """
        with self._pending_lock:
            self._pending += 1
            self._all_done.clear()

        with self._queue_lock:
            heapq.heappush(self._queue, (z, self._queue_seq, task_name))
            self._queue_seq += 1

        for proc in self._processors:
            proc.notify()
            break

    def _dequeue_task(self) -> tuple:
        """
        _dequeue_task — ClassicContext::NextRoutine analogue.

        Pops the highest-priority (lowest z) entry from the heap and returns
        (task_name, callable).  Returns ("", None) when the queue is empty.
        Called exclusively by AstroProcessor._run() — no external callers.
        """
        with self._queue_lock:
            if not self._queue:
                return ("", None)
            z, seq, task_name = heapq.heappop(self._queue)

        with self._task_lock:
            func = self._tasks.get(task_name)

        if func is None:
            return ("", None)

        def _wrapped():
            try:
                func()
            finally:
                with self._pending_lock:
                    self._pending -= 1
                    if self._pending <= 0:
                        self._pending = 0
                        self._all_done.set()

        return (task_name, _wrapped)

    # ── run_until_done ────────────────────────────────────────────────────────

    def run_until_done(self, timeout: float = 10.0) -> bool:
        """
        Block until all currently queued tasks have completed.
        Mirrors the epoch-completion wait in the CyberRT node spin loop.
        Returns True when all tasks finished within timeout, False on timeout.
        """
        return self._all_done.wait(timeout=timeout)

    # ── CheckSchedStatus ──────────────────────────────────────────────────────

    def check_sched_status(self) -> str:
        """
        CheckSchedStatus() — mirrors Scheduler::CheckSchedStatus().

        Builds a snapshot string identical in format to the C++ version:
            proc_id:routine_name:elapsed_ms, …, timestamp: <ns>
        Returns the snapshot string (also emits [ASTRO-SCHED] debug log).
        """
        now_mono = time.monotonic()
        now_ns = int(now_mono * 1e9)
        parts = []
        for proc in self._processors:
            snap = proc.snapshot
            if snap.execute_start_mono > 0.0:
                elapsed_ms = int((now_mono - snap.execute_start_mono) * 1000)
                parts.append(f"{snap.processor_id}:{snap.routine_name}:{elapsed_ms}")
            else:
                parts.append(f"{snap.processor_id}:idle")
        snap_info = ", ".join(parts) + f", timestamp: {now_ns}"
        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHEDULER] CheckSchedStatus epoch={self._epoch_index} "
             f"procs={len(self._processors)} snap={snap_info}")
        return snap_info

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Shutdown() — mirrors Scheduler::Shutdown().

        Sets stop flag, stops all processors, drains the task registry,
        shuts down the ThreadPoolExecutor.
        """
        if self._stop:
            return
        self._stop = True

        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHEDULER] Shutdown draining {len(self._processors)} processors")

        for proc in self._processors:
            proc.stop()

        with self._task_lock:
            self._tasks.clear()

        with self._queue_lock:
            self._queue.clear()

        self._executor.shutdown(wait=False)
        _dbg("ASTRO-SCHED", "AstroScheduler shutdown complete")


@dataclasses.dataclass
class ChoreographyTaskConf:
    """Per-task choreography conf — mirrors proto::ChoreographyTask."""
    name:         str = ""
    priority:     int = 0
    processor_id: int = 0


class _ChoreographyProcessor:
    """Dedicated processor for one choreography slot."""

    def __init__(self, proc_id: str, scheduler: "AstroChoreographyScheduler",
                 slot_idx: int) -> None:
        self._id = proc_id; self._sched = scheduler; self._slot_idx = slot_idx
        self._running = False; self._wake = threading.Event()
        self._snap = AstroSnapshot(processor_id=proc_id)
        self._future: Optional[concurrent.futures.Future] = None

    def bind_and_start(self, ex: concurrent.futures.ThreadPoolExecutor) -> None:
        if self._running: return
        self._running = True; self._future = ex.submit(self._run)

    def stop(self) -> None:
        if not self._running: return
        self._running = False; self._wake.set()
        if self._future:
            try: self._future.result(timeout=2.0)
            except Exception: pass

    @property
    def snapshot(self) -> AstroSnapshot: return self._snap

    def notify(self) -> None: self._wake.set()

    def _run(self) -> None:
        _dbg("ASTRO-PROC", f"[ASTRO-PROCESSOR] ChoreographyProcessor::Run id={self._id} ONLINE")
        while self._running:
            task_name, func = self._sched._dequeue_chore(self._slot_idx)
            if func is not None:
                self._snap.execute_start_mono = time.monotonic()
                self._snap.routine_name = task_name
                _dbg("ASTRO-PROC", f"[ASTRO-PROCESSOR] chore tick: cell='{task_name}' proc={self._id}")
                try: func()
                except Exception as exc:
                    _dbg("ASTRO-PROC", f"chore exc cell={task_name} exc={exc}")
                finally: self._snap.execute_start_mono = 0.0; self._snap.routine_name = ""
            else:
                self._wake.clear(); self._wake.wait(timeout=0.01)


class _PoolProcessor:
    """Pool processor — shares the scheduler's pool heapq."""

    def __init__(self, proc_id: str, scheduler: "AstroChoreographyScheduler") -> None:
        self._id = proc_id; self._sched = scheduler
        self._running = False; self._wake = threading.Event()
        self._snap = AstroSnapshot(processor_id=proc_id)
        self._future: Optional[concurrent.futures.Future] = None

    def bind_and_start(self, ex: concurrent.futures.ThreadPoolExecutor) -> None:
        if self._running: return
        self._running = True; self._future = ex.submit(self._run)

    def stop(self) -> None:
        if not self._running: return
        self._running = False; self._wake.set()
        if self._future:
            try: self._future.result(timeout=2.0)
            except Exception: pass

    @property
    def snapshot(self) -> AstroSnapshot: return self._snap

    def notify(self) -> None: self._wake.set()

    def _run(self) -> None:
        _dbg("ASTRO-PROC", f"[ASTRO-PROCESSOR] PoolProcessor::Run id={self._id} ONLINE")
        while self._running:
            task_name, func = self._sched._dequeue_pool()
            if func is not None:
                self._snap.execute_start_mono = time.monotonic()
                self._snap.routine_name = task_name
                _dbg("ASTRO-PROC", f"[ASTRO-PROCESSOR] pool tick: cell='{task_name}' proc={self._id}")
                try: func()
                except Exception as exc:
                    _dbg("ASTRO-PROC", f"pool exc cell={task_name} exc={exc}")
                finally: self._snap.execute_start_mono = 0.0; self._snap.routine_name = ""
            else:
                self._wake.clear(); self._wake.wait(timeout=0.01)


class AstroChoreographyScheduler:
    """
    Choreography-policy scheduler — Python port of SchedulerChoreography.

    Tasks in *chore_task_confs* → pinned choreography processors.
    All other tasks → shared pool heapq served by pool processors.

    Usage::

        confs = [ChoreographyTaskConf("self_attn", priority=0, processor_id=0)]
        sched = AstroChoreographyScheduler(chore_task_confs=confs,
                                           num_chore_processors=1,
                                           num_pool_processors=1)
        sched.create_task(lambda: render("self_attn"), "self_attn", z=0)
        sched.create_task(lambda: render("norm"),      "norm",      z=3)
        sched.run_until_done()
        sched.shutdown()
    """

    def __init__(
        self,
        chore_task_confs:              Optional[List[ChoreographyTaskConf]] = None,
        num_chore_processors:          int = 1,
        num_pool_processors:           int = 1,
        choreography_affinity:         str = "",
        pool_affinity:                 str = "",
        choreography_processor_policy: str = "SCHED_OTHER",
        pool_processor_policy:         str = "SCHED_OTHER",
        choreography_processor_prio:   int = 0,
        pool_processor_prio:           int = 0,
        choreography_cpuset:           Optional[List[int]] = None,
        pool_cpuset:                   Optional[List[int]] = None,
    ) -> None:
        self._stop: bool = False

        # cr_confs_ — mirrors SchedulerChoreography::cr_confs_
        self._cr_confs: Dict[str, ChoreographyTaskConf] = {
            c.name: c for c in (chore_task_confs or [])
        }

        # OS-level policy fields (stored; not applied in Python)
        self.choreography_affinity         = choreography_affinity
        self.pool_affinity                 = pool_affinity
        self.choreography_processor_policy = choreography_processor_policy
        self.pool_processor_policy         = pool_processor_policy
        self.choreography_processor_prio   = choreography_processor_prio
        self.pool_processor_prio           = pool_processor_prio
        self.choreography_cpuset: List[int] = choreography_cpuset or []
        self.pool_cpuset:         List[int] = pool_cpuset or []

        self._tasks: Dict[str, Any]   = {}
        self._task_lock               = threading.Lock()

        self._num_chore_procs = max(1, num_chore_processors)
        self._chore_proc_queues: List[List[tuple]] = [[] for _ in range(self._num_chore_procs)]
        self._chore_queue_locks: List[threading.Lock] = [threading.Lock() for _ in range(self._num_chore_procs)]
        self._chore_map: Dict[str, int] = {}

        self._num_pool_procs  = max(1, num_pool_processors)
        self._pool_queue: List[tuple] = []
        self._pool_q_lock = threading.Lock()
        self._pool_q_seq  = 0

        self._pending     = 0
        self._pending_lock = threading.Lock()
        self._all_done    = threading.Event()
        self._all_done.set()

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._num_chore_procs + self._num_pool_procs,
            thread_name_prefix="astro_chore",
        )
        self._chore_processors: List[_ChoreographyProcessor] = []
        self._pool_processors:  List[_PoolProcessor]          = []
        self._create_processors()

        _dbg("ASTRO-CHORE",
             f"AstroChoreographyScheduler constructed "
             f"chore_procs={self._num_chore_procs} pool_procs={self._num_pool_procs} "
             f"chore_tasks={list(self._cr_confs.keys())}")

    def _create_processors(self) -> None:
        """CreateProcessor() — mirrors SchedulerChoreography::CreateProcessor()."""
        for idx in range(self._num_chore_procs):
            p = _ChoreographyProcessor(f"chore_{idx}", self, idx)
            p.bind_and_start(self._executor)
            self._chore_processors.append(p)
            _dbg("ASTRO-CHORE",
                 f"CreateProcessor: chore idx={idx} ONLINE "
                 f"policy={self.choreography_processor_policy} prio={self.choreography_processor_prio}")
        for idx in range(self._num_pool_procs):
            p = _PoolProcessor(f"pool_{idx}", self)
            p.bind_and_start(self._executor)
            self._pool_processors.append(p)
            _dbg("ASTRO-CHORE",
                 f"CreateProcessor: pool  idx={idx} ONLINE "
                 f"policy={self.pool_processor_policy} prio={self.pool_processor_prio}")

    def create_task(self, func: Callable[[], None], task_name: str,
                    z: int = 3, channel_path: Optional[str] = None) -> bool:
        """DispatchTask — route to choreography or pool based on cr_confs_."""
        if self._stop: return False
        with self._task_lock:
            self._tasks[task_name] = func
        if task_name in self._cr_confs:
            conf    = self._cr_confs[task_name]
            proc_id = conf.processor_id % self._num_chore_procs
            self._chore_map[task_name] = proc_id
            self._dispatch_chore(task_name, conf.priority, proc_id)
            _dbg("ASTRO-CHORE", f"DispatchTask CHORE task={task_name} proc={proc_id} prio={conf.priority}")
        else:
            self._dispatch_pool(task_name, z)
            _dbg("ASTRO-CHORE", f"DispatchTask POOL  task={task_name} z={z}")
        if channel_path is not None:
            def _notify_cb() -> None:
                if not self._stop: self.notify_processor(task_name, z)
            DataNotifier.instance().add_notifier(channel_path, Notifier(_notify_cb))
        return True

    def notify_processor(self, task_name: str, z: int = 3) -> bool:
        """NotifyProcessor — mirrors SchedulerChoreography::NotifyProcessor(crid)."""
        if self._stop: return False
        with self._task_lock:
            if task_name not in self._tasks: return False
        if task_name in self._chore_map:
            conf = self._cr_confs.get(task_name)
            self._dispatch_chore(task_name, conf.priority if conf else 0, self._chore_map[task_name])
        else:
            self._dispatch_pool(task_name, z)
        return True

    def remove_task(self, task_name: str) -> bool:
        """RemoveTask — mirrors SchedulerChoreography::RemoveTask."""
        with self._task_lock:
            removed = self._tasks.pop(task_name, None) is not None
        self._cr_confs.pop(task_name, None)
        self._chore_map.pop(task_name, None)
        return removed

    def remove_croutine(self, task_name: str) -> bool:
        """RemoveCRoutine — mirrors SchedulerChoreography::RemoveCRoutine(crid)."""
        return self.remove_task(task_name)

    def _dispatch_chore(self, task_name: str, priority: int, proc_id: int) -> None:
        with self._pending_lock:
            self._pending += 1; self._all_done.clear()
        with self._chore_queue_locks[proc_id]:
            heapq.heappush(self._chore_proc_queues[proc_id], (priority, id(task_name), task_name))
        if proc_id < len(self._chore_processors):
            self._chore_processors[proc_id].notify()

    def _dispatch_pool(self, task_name: str, z: int) -> None:
        with self._pending_lock:
            self._pending += 1; self._all_done.clear()
        with self._pool_q_lock:
            heapq.heappush(self._pool_queue, (z, self._pool_q_seq, task_name))
            self._pool_q_seq += 1
        for p in self._pool_processors: p.notify(); break

    def _dequeue_chore(self, proc_id: int) -> Tuple[str, Optional[Callable]]:
        with self._chore_queue_locks[proc_id]:
            if not self._chore_proc_queues[proc_id]: return "", None
            _, _, task_name = heapq.heappop(self._chore_proc_queues[proc_id])
        with self._task_lock:
            func = self._tasks.get(task_name)
        return ("", None) if func is None else (task_name, self._wrap(func))

    def _dequeue_pool(self) -> Tuple[str, Optional[Callable]]:
        with self._pool_q_lock:
            if not self._pool_queue: return "", None
            _, _, task_name = heapq.heappop(self._pool_queue)
        with self._task_lock:
            func = self._tasks.get(task_name)
        return ("", None) if func is None else (task_name, self._wrap(func))

    def _wrap(self, func: Callable) -> Callable:
        def _wrapped() -> None:
            try: func()
            finally:
                with self._pending_lock:
                    self._pending = max(0, self._pending - 1)
                    if self._pending == 0: self._all_done.set()
        return _wrapped

    def run_until_done(self, timeout: float = 10.0) -> bool:
        return self._all_done.wait(timeout=timeout)

    def check_sched_status(self) -> str:
        """CheckSchedStatus — mirrors SchedulerChoreography status snapshot."""
        now = time.monotonic(); parts: List[str] = []
        for p in self._chore_processors + self._pool_processors:  # type: ignore[operator]
            s = p.snapshot
            if s.execute_start_mono > 0.0:
                parts.append(f"{s.processor_id}:{s.routine_name}:{int((now-s.execute_start_mono)*1000)}ms")
            else:
                parts.append(f"{s.processor_id}:idle")
        snap = ", ".join(parts) + f", timestamp: {int(now*1e9)}"
        _dbg("ASTRO-CHORE", f"[ASTRO-CHORE] CheckSchedStatus snap={snap}")
        return snap

    def shutdown(self) -> None:
        """Shutdown — mirrors SchedulerChoreography::Shutdown()."""
        if self._stop: return
        self._stop = True
        _dbg("ASTRO-CHORE",
             f"[ASTRO-CHORE] Shutdown chore={len(self._chore_processors)} pool={len(self._pool_processors)}")
        for p in self._chore_processors: p.stop()
        for p in self._pool_processors:  p.stop()
        with self._task_lock: self._tasks.clear()
        for q in self._chore_proc_queues: q.clear()
        with self._pool_q_lock: self._pool_queue.clear()
        self._executor.shutdown(wait=False)
        _dbg("ASTRO-CHORE", "AstroChoreographyScheduler shutdown complete")


class AstroSchedulerFactory:
    """Port of SchedulerFactory — creates Classic or Choreography scheduler."""
    @staticmethod
    def create(policy: str = "classic", **kwargs):
        if policy == "choreography":
            return AstroChoreographyScheduler(**kwargs)
        return AstroScheduler(**kwargs)


class AstroChoreographyContext:
    """Port of ChoreographyContext — DAG execution context."""
    def __init__(self):
        self._tasks: list = []
        self._running = False
    def add_task(self, name: str, func, priority: int = 0):
        self._tasks.append({"name": name, "func": func, "priority": priority})
        self._tasks.sort(key=lambda t: -t["priority"])
    def execute(self):
        self._running = True
        for t in self._tasks:
            if not self._running: break
            t["func"]()
    def stop(self): self._running = False


class AstroCvWrapper:
    """Port of CvWrapper — condition variable wrapper."""
    def __init__(self):
        self._cv = _cv_threading.Condition()
    def notify_one(self):
        with self._cv: self._cv.notify()
    def notify_all(self):
        with self._cv: self._cv.notify_all()
    def wait(self, timeout=None):
        with self._cv: self._cv.wait(timeout)


class AstroMutexWrapper:
    """Port of MutexWrapper — read-write lock."""
    def __init__(self):
        self._lock = _cv_threading.RLock()
    def lock(self): self._lock.acquire()
    def unlock(self): self._lock.release()
    def __enter__(self): self._lock.acquire(); return self
    def __exit__(self, *a): self._lock.release()


class AstroPinThread:
    """Port of PinThread — CPU affinity (no-op in Python, for interface compat)."""
    @staticmethod
    def pin_to_cpu(cpu_id: int): pass  # os.sched_setaffinity not portable
    @staticmethod
    def set_name(name: str):
        try: _cv_threading.current_thread().name = name
        except: pass


class AstroProcessorContext:
    """Port of ProcessorContext — base class for processor execution contexts."""
    def __init__(self):
        self._shutdown = False
        self._notified = _cv_threading.Event()
    def shutdown(self): self._shutdown = True; self._notified.set()
    def wait(self, timeout=1.0): self._notified.wait(timeout); self._notified.clear()
    def notify(self): self._notified.set()


__all__ = [
    "AstroChoreographyContext",
    "AstroChoreographyScheduler",
    "AstroCvWrapper",
    "AstroMutexWrapper",
    "AstroPinThread",
    "AstroProcessor",
    "AstroProcessorContext",
    "AstroScheduler",
    "AstroSchedulerFactory",
    "AstroSnapshot",
    "ChoreographyTaskConf",
    "LoopScheduler",
    "_ChoreographyProcessor",
    "_PoolProcessor",
]
