import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)

class AstroSubscriberListener:
    """Port of SubscriberListener — callback on subscription match."""
    def __init__(self):
        self._on_match_callbacks: list = []
    def on_subscription_matched(self, channel_id: str, matched: bool):
        for cb in self._on_match_callbacks: cb(channel_id, matched)
    def add_callback(self, cb): self._on_match_callbacks.append(cb)

# --- service_discovery/communication/participant_listener.h (53 lines) ---


# --- service_discovery/communication/participant_listener.h (53 lines) ---
class AstroParticipantListener:
    """Port of ParticipantListener — callback on participant discovery."""
    def __init__(self):
        self._on_discovery: list = []
    def on_participant_discovery(self, participant_id: str, joined: bool):
        for cb in self._on_discovery: cb(participant_id, joined)
    def add_callback(self, cb): self._on_discovery.append(cb)

# --- transport/shm/multicast_notifier.h (58 lines) ---

