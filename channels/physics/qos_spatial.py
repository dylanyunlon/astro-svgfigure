"""
channels/physics/qos_spatial.py
Apollo CyberRT QoS profile → spatial physics parameter mapping.

Upstream reference: apollo-cyber/transport/qos/qos_profile_conf.cc
Lifts QoS constraints from the communication layer into a physical
simulation space for pub/sub particle rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# QoS enumerations (mirrors qos_profile_conf.cc types)
# ---------------------------------------------------------------------------

RELIABLE    = "RELIABLE"
BEST_EFFORT = "BEST_EFFORT"

VOLATILE       = "VOLATILE"
TRANSIENT_LOCAL = "TRANSIENT_LOCAL"

KEEP_LAST = "KEEP_LAST"
KEEP_ALL  = "KEEP_ALL"

# ---------------------------------------------------------------------------
# Apollo CyberRT QoS profiles (exact values from qos_profile_conf.cc)
# Profile rank order determines priority (0 = lowest → 7 = highest).
# ---------------------------------------------------------------------------

APOLLO_PROFILES: dict[str, dict[str, Any]] = {
    # rank 0 – generic default, minimal buffering
    "DEFAULT": {
        "depth":       1,
        "mps":         0,
        "reliability": RELIABLE,
        "durability":  VOLATILE,
        "history":     KEEP_LAST,
        "rank":        0,
    },
    # rank 1 – high-frequency sensor streams, loss-tolerant
    "SENSOR_DATA": {
        "depth":       5,
        "mps":         0,
        "reliability": BEST_EFFORT,
        "durability":  VOLATILE,
        "history":     KEEP_LAST,
        "rank":        1,
    },
    # rank 2 – large parameter stores, must not lose messages
    "PARAMETERS": {
        "depth":       1000,
        "mps":         0,
        "reliability": RELIABLE,
        "durability":  VOLATILE,
        "history":     KEEP_LAST,
        "rank":        2,
    },
    # rank 3 – service calls, durable for late-joiners
    "SERVICES_DEFAULT": {
        "depth":       10,
        "mps":         0,
        "reliability": RELIABLE,
        "durability":  TRANSIENT_LOCAL,
        "history":     KEEP_LAST,
        "rank":        3,
    },
    # rank 4 – parameter change events, large queue
    "PARAM_EVENT": {
        "depth":       1000,
        "mps":         0,
        "reliability": RELIABLE,
        "durability":  VOLATILE,
        "history":     KEEP_LAST,
        "rank":        4,
    },
    # rank 5 – system-wide coordination, unlimited depth
    "SYSTEM_DEFAULT": {
        "depth":       0,
        "mps":         0,
        "reliability": RELIABLE,
        "durability":  TRANSIENT_LOCAL,
        "history":     KEEP_LAST,
        "rank":        5,
    },
    # rank 6 – static transforms, keep entire history
    "TF_STATIC": {
        "depth":       10,
        "mps":         0,
        "reliability": RELIABLE,
        "durability":  TRANSIENT_LOCAL,
        "history":     KEEP_ALL,
        "rank":        6,
    },
    # rank 7 – topology change broadcasts, keep entire history
    "TOPO_CHANGE": {
        "depth":       10,
        "mps":         0,
        "reliability": RELIABLE,
        "durability":  TRANSIENT_LOCAL,
        "history":     KEEP_ALL,
        "rank":        7,
    },
}

_MAX_RANK = max(p["rank"] for p in APOLLO_PROFILES.values())  # 7

# ---------------------------------------------------------------------------
# Mapping constants
# ---------------------------------------------------------------------------

_VISCOSITY: dict[str, float] = {
    RELIABLE:    0.020,   # cohesive – particles cling together
    BEST_EFFORT: 0.001,   # scatter  – particles disperse freely
}

_BOUNDARY_FRICTION: dict[str, float] = {
    TRANSIENT_LOCAL: 0.95,  # sticky wall – messages linger at boundary
    VOLATILE:        0.30,  # elastic     – messages bounce away
}

_TRAIL_LENGTH_CAP   = 30    # hard cap on visual trail length
_MPS_UNLIMITED_RATE = 120   # particle emitter rate when mps == 0 (unlimited)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def qos_to_physics(profile: str | dict[str, Any]) -> dict[str, float | int]:
    """Convert an Apollo QoS profile to spatial physics parameters.

    Parameters
    ----------
    profile:
        Either a profile name string (key in ``APOLLO_PROFILES``) or a raw
        QoS dict with keys ``reliability``, ``durability``, ``depth``,
        ``mps``, and ``rank``.

    Returns
    -------
    dict with keys:
        viscosity        – particle cohesion coefficient  [0.001 … 0.020]
        boundary_friction – wall-stickiness coefficient   [0.30  … 0.95]
        trail_length     – discrete trail step count      [0 … 30]
        emitter_rate     – particles emitted per second   [≥1]
        force_multiplier – net force scale on particles   [1.0 … 4.5]
    """
    if isinstance(profile, str):
        key = profile.upper()
        if key not in APOLLO_PROFILES:
            raise KeyError(
                f"Unknown Apollo QoS profile {profile!r}. "
                f"Valid names: {sorted(APOLLO_PROFILES)}"
            )
        qos = APOLLO_PROFILES[key]
    elif isinstance(profile, dict):
        qos = profile
    else:
        raise TypeError(
            f"profile must be a str or dict, got {type(profile).__name__!r}"
        )

    reliability = qos.get("reliability", RELIABLE)
    durability  = qos.get("durability",  VOLATILE)
    depth       = int(qos.get("depth", 1))
    mps         = int(qos.get("mps",   0))
    rank        = float(qos.get("rank", 0))

    viscosity         = _VISCOSITY.get(reliability, _VISCOSITY[RELIABLE])
    boundary_friction = _BOUNDARY_FRICTION.get(durability, _BOUNDARY_FRICTION[VOLATILE])
    trail_length      = min(depth, _TRAIL_LENGTH_CAP)
    emitter_rate      = _MPS_UNLIMITED_RATE if mps == 0 else max(1, mps)

    # priority derived from normalised rank: rank ∈ [0, _MAX_RANK]
    priority          = rank / _MAX_RANK if _MAX_RANK > 0 else 0.0
    force_multiplier  = round(1.0 + priority * 0.5, 6)

    return {
        "viscosity":         viscosity,
        "boundary_friction": boundary_friction,
        "trail_length":      trail_length,
        "emitter_rate":      emitter_rate,
        "force_multiplier":  force_multiplier,
    }


def physics_to_qos(physics_params: dict[str, float | int]) -> dict[str, Any]:
    """Approximate reverse mapping: spatial physics → Apollo QoS profile.

    Because the forward mapping is lossy (many QoS fields collapse into one
    physics value) this reconstruction is *approximate*.  The returned dict
    identifies the **closest named profile** together with the inferred raw
    QoS fields.

    Parameters
    ----------
    physics_params:
        Dict with any subset of the keys produced by ``qos_to_physics``.
        Missing keys fall back to the DEFAULT profile's physics values.

    Returns
    -------
    dict with keys:
        profile_name     – name of the closest Apollo profile
        reliability      – RELIABLE | BEST_EFFORT
        durability       – TRANSIENT_LOCAL | VOLATILE
        depth            – estimated history depth (int)
        mps              – estimated messages-per-second cap (int)
        rank             – estimated profile priority rank (int)
    """
    # --- infer discrete QoS fields from physics knobs ---

    # viscosity → reliability
    viscosity  = float(physics_params.get("viscosity", _VISCOSITY[RELIABLE]))
    reliability = (
        RELIABLE    if viscosity >= (_VISCOSITY[RELIABLE] + _VISCOSITY[BEST_EFFORT]) / 2
        else BEST_EFFORT
    )

    # boundary_friction → durability
    bf = float(physics_params.get("boundary_friction", _BOUNDARY_FRICTION[VOLATILE]))
    durability = (
        TRANSIENT_LOCAL
        if bf >= (_BOUNDARY_FRICTION[TRANSIENT_LOCAL] + _BOUNDARY_FRICTION[VOLATILE]) / 2
        else VOLATILE
    )

    # trail_length → depth  (inverse of min(depth, 30))
    trail_length = int(physics_params.get("trail_length", 1))
    depth = trail_length  # best we can do; values > 30 are irrecoverable

    # emitter_rate → mps
    emitter_rate = int(physics_params.get("emitter_rate", _MPS_UNLIMITED_RATE))
    mps = 0 if emitter_rate >= _MPS_UNLIMITED_RATE else emitter_rate

    # force_multiplier → rank
    fm       = float(physics_params.get("force_multiplier", 1.0))
    priority = max(0.0, min(1.0, (fm - 1.0) / 0.5))
    rank     = round(priority * _MAX_RANK)

    # --- find the closest named profile by scoring ---
    inferred = {
        "reliability": reliability,
        "durability":  durability,
        "depth":       depth,
        "mps":         mps,
        "rank":        rank,
    }

    best_name  = "DEFAULT"
    best_score = -1

    for name, qos in APOLLO_PROFILES.items():
        score = 0
        if qos["reliability"] == reliability:
            score += 3   # high weight – drives viscosity
        if qos["durability"]  == durability:
            score += 3   # high weight – drives boundary_friction
        if qos["rank"]        == rank:
            score += 2
        if min(qos["depth"], _TRAIL_LENGTH_CAP) == trail_length:
            score += 1
        if (qos["mps"] == 0 and mps == 0) or qos["mps"] == mps:
            score += 1
        if score > best_score:
            best_score = score
            best_name  = name

    return {"profile_name": best_name, **inferred}


# ---------------------------------------------------------------------------
# Convenience: pre-compute physics for every profile
# ---------------------------------------------------------------------------

PROFILE_PHYSICS: dict[str, dict[str, float | int]] = {
    name: qos_to_physics(name) for name in APOLLO_PROFILES
}


# ---------------------------------------------------------------------------
# Quick self-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 64)
    print("Apollo QoS → Physics mappings")
    print("=" * 64)
    for name, phys in PROFILE_PHYSICS.items():
        print(f"\n{name}")
        for k, v in phys.items():
            print(f"  {k:<22} {v}")

    print("\n" + "=" * 64)
    print("Round-trip: physics → QoS (approximate)")
    print("=" * 64)
    for name, phys in PROFILE_PHYSICS.items():
        recovered = physics_to_qos(phys)
        match = "✓" if recovered["profile_name"] == name else "≈"
        print(f"  {match} {name:<18} → {recovered['profile_name']}")
