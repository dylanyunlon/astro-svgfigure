import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)


# --- transport/rtps/sub_listener.h (65 lines) ---
class AstroSubListener:
    """Port of SubListener — RTPS subscription data listener."""
    def __init__(self, callback=None):
        self._callback = callback
    def on_data_available(self, data: dict, info: dict = None):
        if self._callback: self._callback(data, info or {})

# --- transport/rtps/attributes_filler.h (54 lines) ---

