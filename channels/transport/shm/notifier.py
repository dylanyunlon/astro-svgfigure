import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)


# --- transport/shm/multicast_notifier.h (58 lines) ---
class AstroMulticastNotifier:
    """Port of MulticastNotifier — multicast-based notification (simplified to callback)."""
    def __init__(self):
        self._listeners: list = []
    def notify(self, info):
        for listener in self._listeners: listener(info)
    def add_listener(self, cb): self._listeners.append(cb)
    def shutdown(self): self._listeners.clear()

# --- transport/shm/notifier_base.h (45 lines) ---


# --- transport/shm/notifier_base.h (45 lines) ---
class AstroNotifierBase:
    """Port of NotifierBase — abstract notifier interface."""
    def notify(self, info) -> bool: raise NotImplementedError
    def listen(self, timeout: float = 1.0): raise NotImplementedError
    def shutdown(self): pass

# --- transport/shm/notifier_factory.h (42 lines) ---


# --- transport/shm/notifier_factory.h (42 lines) ---
class AstroNotifierFactory:
    """Port of NotifierFactory — creates ConditionNotifier or MulticastNotifier."""
    @staticmethod
    def create(mode: str = "condition"):
        if mode == "multicast": return AstroMulticastNotifier()
        return AstroConditionNotifier()

# --- transport/shm/posix_segment.h (49 lines) ---

