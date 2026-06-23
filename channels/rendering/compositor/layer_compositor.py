
# ─────────────────────────────────────────────────────────────────────────────
# M826: Layer Compositor — z-layer compositing + blend modes + AOV output
# ─────────────────────────────────────────────────────────────────────────────

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class BlendMode(Enum):
    NORMAL = "normal"
    ADD = "add"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"


def blend_normal(src: Tuple, dst: Tuple, alpha: float) -> Tuple:
    return tuple(src[i] * alpha + dst[i] * (1 - alpha) for i in range(3))

def blend_add(src: Tuple, dst: Tuple, alpha: float) -> Tuple:
    return tuple(min(1.0, dst[i] + src[i] * alpha) for i in range(3))

def blend_multiply(src: Tuple, dst: Tuple, alpha: float) -> Tuple:
    mult = tuple(src[i] * dst[i] for i in range(3))
    return tuple(mult[i] * alpha + dst[i] * (1 - alpha) for i in range(3))

def blend_screen(src: Tuple, dst: Tuple, alpha: float) -> Tuple:
    scr = tuple(1 - (1 - src[i]) * (1 - dst[i]) for i in range(3))
    return tuple(scr[i] * alpha + dst[i] * (1 - alpha) for i in range(3))

BLEND_FUNCTIONS = {
    BlendMode.NORMAL: blend_normal,
    BlendMode.ADD: blend_add,
    BlendMode.MULTIPLY: blend_multiply,
    BlendMode.SCREEN: blend_screen,
}


@dataclass
class CompositorLayer:
    """A single compositing layer."""
    name: str
    z_order: int
    blend_mode: BlendMode = BlendMode.NORMAL
    opacity: float = 1.0
    visible: bool = True
    content: Any = None  # rendered pass output


class LayerCompositor:
    """
    Composites multiple render layers by z-order using specified blend modes.

    [ASTRO-COMPOSITOR] Back-to-front painter's algorithm with per-layer blend.

    Layers:
      - background (z=0): gradient or solid color
      - edges (z=1): spline edges between cells
      - cells_body (z=2-8): cell rectangles by z_layer
      - cells_icon (z=3-9): species SDF icons
      - effects (z=10): bloom, glow, godray
      - overlay (z=11): debug, selection ring, HUD
    """

    def __init__(self):
        self._layers: Dict[str, CompositorLayer] = {}

    def add_layer(self, name: str, z_order: int,
                  blend_mode: BlendMode = BlendMode.NORMAL,
                  opacity: float = 1.0) -> CompositorLayer:
        layer = CompositorLayer(name=name, z_order=z_order,
                                blend_mode=blend_mode, opacity=opacity)
        self._layers[name] = layer
        print(f"[ASTRO-COMPOSITOR] add layer: {name} z={z_order} "
              f"blend={blend_mode.value}", file=sys.stderr)
        return layer

    def set_content(self, name: str, content: Any):
        if name in self._layers:
            self._layers[name].content = content

    def composite(self, width: int = 1920, height: int = 1080) -> dict:
        """
        Composite all visible layers back-to-front.
        Returns composite result descriptor.
        """
        sorted_layers = sorted(
            [l for l in self._layers.values() if l.visible],
            key=lambda l: l.z_order)

        result = {
            "width": width, "height": height,
            "layers_composited": [],
            "total_layers": len(sorted_layers),
        }

        for layer in sorted_layers:
            blend_fn = BLEND_FUNCTIONS.get(layer.blend_mode, blend_normal)
            result["layers_composited"].append({
                "name": layer.name,
                "z": layer.z_order,
                "blend": layer.blend_mode.value,
                "opacity": layer.opacity,
            })
            print(f"[ASTRO-COMPOSITOR] composite: {layer.name} "
                  f"z={layer.z_order} blend={layer.blend_mode.value} "
                  f"opacity={layer.opacity:.2f}", file=sys.stderr)

        return result

    def get_sorted_layers(self) -> List[CompositorLayer]:
        return sorted(
            [l for l in self._layers.values() if l.visible],
            key=lambda l: l.z_order)


class AOVOutput:
    """
    Arbitrary Output Variables — separate render passes for inspection.

    [ASTRO-COMPOSITOR] AOVs let you inspect individual render channels:
      - color: final composited RGB
      - depth: z-layer depth map
      - species_id: per-pixel species classification
      - normal: surface normal (for 3D-lit cells)
      - velocity: motion vectors (for temporal effects)
    """

    AOV_CHANNELS = ["color", "depth", "species_id", "normal", "velocity", "opacity"]

    def __init__(self):
        self._buffers: Dict[str, Any] = {}
        self._enabled: Dict[str, bool] = {ch: False for ch in self.AOV_CHANNELS}
        self._enabled["color"] = True  # always enabled

    def enable(self, channel: str):
        if channel in self._enabled:
            self._enabled[channel] = True
            print(f"[ASTRO-COMPOSITOR] AOV enabled: {channel}", file=sys.stderr)

    def disable(self, channel: str):
        if channel != "color" and channel in self._enabled:
            self._enabled[channel] = False

    def write(self, channel: str, data: Any):
        if self._enabled.get(channel, False):
            self._buffers[channel] = data

    def read(self, channel: str) -> Optional[Any]:
        return self._buffers.get(channel)

    @property
    def active_channels(self) -> List[str]:
        return [ch for ch, en in self._enabled.items() if en]
