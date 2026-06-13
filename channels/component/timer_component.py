import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar
from channels.component.component_base import AstroComponentBase

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════════════

# --- scheduler/scheduler_factory.h (40 lines) ---

# --- scheduler/policy/choreography_context.h (63 lines) ---

# --- scheduler/common/cv_wrapper.h (40 lines) ---

# --- scheduler/common/mutex_wrapper.h (40 lines) ---

# --- scheduler/common/pin_thread.h (43 lines) ---

# --- scheduler/processor_context.h (48 lines) ---

# --- component/timer_component.h (68 lines) ---
class AstroTimerComponent(AstroComponentBase):
    """Port of TimerComponent — fires Proc() at fixed interval."""
    def __init__(self, interval_ms: int = 100, **kwargs):
        super().__init__(**kwargs)
        self._interval = interval_ms / 1000.0
        self._timer = None
        self._running = False
    def initialize(self) -> bool:
        self._running = True
        self._schedule()
        return True
    def _schedule(self):
        if self._running:
            self._timer = _cv_threading.Timer(self._interval, self._tick)
            self._timer.daemon = True
            self._timer.start()
    def _tick(self):
        if self._running:
            self.process()
            self._schedule()
    def clear(self):
        self._running = False
        if self._timer: self._timer.cancel()

# --- data/data_visitor_base.h (55 lines) ---
import abc as _dvb_abc

