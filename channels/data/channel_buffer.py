import json
import os
from typing import Any, List, Optional

from channels.data.notifier import DataNotifier, _dbg

CHANNELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ChannelBuffer:
    """Mirrors CacheBuffer<shared_ptr<T>> — here it's a file-backed JSON slot."""
    __slots__ = ("channel_id", "queue", "max_size")

    def __init__(self, channel_id: str, max_size: int = 1):
        self.channel_id = channel_id
        self.queue: List[Any] = []
        self.max_size = max_size

    def fill(self, data: Any):
        self.queue.append(data)
        if len(self.queue) > self.max_size:
            self.queue.pop(0)
        # Write to file — the "shared memory" equivalent
        full = os.path.join(CHANNELS_DIR, self.channel_id)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            if isinstance(data, str):
                f.write(data)
            else:
                json.dump(data, f, indent=2)

    def latest(self) -> Optional[Any]:
        return self.queue[-1] if self.queue else None
