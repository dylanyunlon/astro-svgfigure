import os
from typing import Callable, Dict, List, Optional

CHANNELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERBOSE = os.environ.get("ASTRO_NOTIFY_VERBOSE", "0") == "1"


def _dbg(prefix: str, msg: str):
    if VERBOSE:
        print(f"[{prefix}] {msg}")


class Notifier:
    """Single notifier entry — mirrors struct Notifier { callback; }"""
    __slots__ = ("callback",)
    def __init__(self, callback: Callable[[], None]):
        self.callback = callback


class DataNotifier:
    """
    Port of apollo::cyber::data::DataNotifier.
    AddNotifier registers a callback on a channel path.
    Notify fires all callbacks registered on that channel.
    """
    _instance: Optional["DataNotifier"] = None

    def __init__(self):
        self._notifies_map: Dict[str, List[Notifier]] = {}
        self._mtime_cache: Dict[str, float] = {}
        _dbg("ASTRO-NOTIFY", "DataNotifier singleton constructed")

    @classmethod
    def instance(cls) -> "DataNotifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def add_notifier(self, channel_id: str, notifier: Notifier):
        """AddNotifier — register callback on channel. Mirrors data_notifier.h AddNotifier."""
        if channel_id in self._notifies_map:
            self._notifies_map[channel_id].append(notifier)
            _dbg("ASTRO-NOTIFY", f"ch={channel_id} notifiers={len(self._notifies_map[channel_id])} op=add_existing ok=1")
        else:
            self._notifies_map[channel_id] = [notifier]
            _dbg("ASTRO-NOTIFY", f"ch={channel_id} notifiers=1 op=add_new ok=1")

    def notify(self, channel_id: str) -> bool:
        """Notify — fire all callbacks on channel. Mirrors data_notifier.h Notify."""
        notifiers = self._notifies_map.get(channel_id)
        if notifiers:
            _dbg("ASTRO-NOTIFY", f"ch={channel_id} notifiers={len(notifiers)} op=notify ok=1")
            for n in notifiers:
                if n.callback:
                    n.callback()
            return True
        _dbg("ASTRO-NOTIFY", f"ch={channel_id} notifiers=0 op=notify_miss ok=0")
        return False

    def check_mtime(self, channel_id: str) -> bool:
        """ASTRO extension: check if file changed since last check. Not in Apollo original."""
        full_path = os.path.join(CHANNELS_DIR, channel_id)
        if not os.path.exists(full_path):
            return False
        mtime = os.path.getmtime(full_path)
        old = self._mtime_cache.get(channel_id, 0)
        if mtime > old:
            self._mtime_cache[channel_id] = mtime
            return True
        return False
