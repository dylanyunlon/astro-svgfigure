import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)


# --- transport/rtps/attributes_filler.h (54 lines) ---
class AstroAttributesFiller:
    """Port of AttributesFiller — fills RTPS publisher/subscriber attributes."""
    @staticmethod
    def fill_publisher_attrs(channel_name: str, qos=None) -> dict:
        return {"channel": channel_name, "reliability": "RELIABLE", "history_depth": 10, **(qos or {})}
    @staticmethod
    def fill_subscriber_attrs(channel_name: str, qos=None) -> dict:
        return {"channel": channel_name, "reliability": "RELIABLE", "history_depth": 10, **(qos or {})}

# --- transport/message/history_attributes.h (45 lines) ---

