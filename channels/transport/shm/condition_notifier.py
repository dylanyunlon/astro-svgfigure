import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



class AstroConditionNotifier:
    """Port of apollo::cyber::transport::ConditionNotifier — condition-variable based notify."""
    import threading as _threading

    def __init__(self):
        self._cond = self._threading.Condition()
        self._readable_infos: list = []
        self._shutdown = False
        _dbg("ASTRO-SHM", "ConditionNotifier created")

    def notify(self, info: AstroReadableInfo) -> bool:
        with self._cond:
            self._readable_infos.append(info)
            self._cond.notify_all()
        _dbg("ASTRO-SHM", f"ConditionNotifier.notify: {info}")
        return True

    def listen(self, timeout: float = 1.0):
        with self._cond:
            if not self._readable_infos:
                self._cond.wait(timeout)
            if self._readable_infos:
                return self._readable_infos.pop(0)
        return None

    def shutdown(self):
        self._shutdown = True
        with self._cond:
            self._cond.notify_all()



