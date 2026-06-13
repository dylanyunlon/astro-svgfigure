from __future__ import annotations
from dataclasses import dataclass, field
import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)









def _bit_invert_if_negative(f: float) -> int:
    """
    Bit-cast float to uint32, then XOR with sign-extension mask.

    Direct port of BitInvertIfNegativeFloat() from MeshDrawCommands.cpp:
        unsigned mask = -int32(f >> 31) | 0x80000000;
        return f ^ mask;

    Converts an IEEE 754 float to a uint32 that preserves the numerical
    ordering under unsigned comparison — used to sort translucent mesh draw
    commands by projected distance without branching.

    鲁迅式：浮点数的符号位是它的立场——
    反转负数的所有位，让它在无符号比较中依然保持正确的大小关系。
    这是一种对不公平规则的巧妙利用：规则不变，解读方式变了。
    """
    raw = _struct.unpack('>I', _struct.pack('>f', f))[0]  # float → uint32 big-endian
    mask = ((-(raw >> 31)) & 0xFFFFFFFF) | 0x80000000
    return (raw ^ mask) & 0xFFFFFFFF


@dataclass


@dataclass


@dataclass


@dataclass



# =============================================================================
# [SceneCaptureRendering] AstroCellCaptureMode + AstroCellCaptureProcessor
# =============================================================================



# =============================================================================
# [ReflectionEnvironmentCapture] AstroCellReflectionCaptureState + pipeline
# =============================================================================



# =============================================================================
# [SceneCaptureRendering] AstroCellCaptureMode + AstroCellCaptureProcessor
# =============================================================================



# =============================================================================
# [ReflectionEnvironmentCapture] AstroCellReflectionCaptureState + pipeline
# =============================================================================

def _clamp_supersample(factor: int) -> int:
    """Clamp supersample factor to [1, 8] — mirrors MinSupersampleCaptureFactor /
    MaxSupersampleCaptureFactor constants in ReflectionEnvironmentCapture.cpp."""
    return max(1, min(8, factor))

