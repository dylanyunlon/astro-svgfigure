import json
import os
from typing import Dict, Optional, Tuple

from channels.data.channel_buffer import ChannelBuffer
from channels.data.data_dispatcher import DataDispatcher
from channels.data.notifier import DataNotifier, _dbg

CHANNELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataVisitor:
    """
    4-channel constraint visitor — ported from DataVisitor<M0,M1,M2,M3>.
    Fuses skeleton + force_field + palette + z_layers into a single Proc() call.
    """
    def __init__(self, cell_id: str):
        self.cell_id = cell_id
        self.channels = {
            "skeleton": f"skeleton/cell/{cell_id}.json",
            "force_field": "physics/force_field.json",
            "palette": "physics/species_assignment.json",
            "z_layers": "physics/z_layers.json",
        }
        self._dispatcher = DataDispatcher.instance()
        self._notifier = DataNotifier.instance()

        # Register buffers — mirrors DataVisitor ctor
        for name, ch_path in self.channels.items():
            buf = ChannelBuffer(ch_path, max_size=2)
            self._dispatcher.add_buffer(buf)
        _dbg("ASTRO-VISITOR", f"ConstraintVisitor ctor: cell={cell_id} binding 4 channels")

    def try_fetch(self) -> Optional[Tuple[dict, dict, dict, dict]]:
        """
        TryFetch — constraint-gated fusion pass.
        Returns (skeleton, force_field, palette, z_layers) or None if any missing.
        Mirrors DataVisitor::TryFetch from data_visitor.h.
        """
        results = {}
        mask = 0
        for i, (name, ch_path) in enumerate(self.channels.items()):
            full = os.path.join(CHANNELS_DIR, ch_path)
            if os.path.exists(full):
                with open(full) as f:
                    results[name] = json.load(f)
                mask |= (1 << i)
            else:
                _dbg("ASTRO-VISITOR", f"TryFetch cell={self.cell_id} missing ch={name}")
                return None

        _dbg("ASTRO-VISITOR", f"TryFetch cell={self.cell_id} mask=0x{mask:x} → fused")
        return (results["skeleton"], results["force_field"],
                results["palette"], results["z_layers"])
