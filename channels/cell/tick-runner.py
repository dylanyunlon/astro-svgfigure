#!/usr/bin/env python3
"""
tick-runner.py — simulates a cell agent tick loop, driven by SKILL.md.

This script plays the role of *every* cell in channels/cell/*/ simultaneously,
one tick at a time, following the physics rules described in
channels/cell/SKILL.md and channels/cell/GEOMETRY_FORMAT.md:

  1. Read channels/cell/*/status.json  → current state of every cell
  2. Read channels/physics/environment.json          → flow field, temperature, gravity
  3. Read channels/physics/cell_lifecycle.json        → energy + lifecycle thresholds
  4. Read channels/physics/species_interaction_matrix.json → attract/repel forces

For each cell, each tick:
  a. energy -= base_consumption ; energy += regeneration_rate (clamped to [0, max_energy])
  b. neighbor interaction force computed from species_interaction_matrix + distance
  c. geometry.json updated: high energy → inflate + smooth surface,
     low energy → shrink + rough surface
  d. energy > division_threshold  → peanut-shaped lobes (division prep)
     energy < apoptosis_threshold → lobes scatter outward, opacity drops
  e. status.json updated: epoch, energy, position, velocity, last_action

geometry.json and status.json are each fully overwritten every tick (per
SKILL.md rule #2: "每个 tick 完整覆盖写 — 不是增量更新").

Usage:
    python3 channels/cell/tick-runner.py --ticks 10
    python3 channels/cell/tick-runner.py --ticks 1 --dt-ms 100 --verbose
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import urllib.error

CELL_DIR = os.path.dirname(os.path.abspath(__file__))
PHYSICS_DIR = os.path.normpath(os.path.join(CELL_DIR, "..", "physics"))

# Server URL for SSE broadcast of geometry updates.
# tick-runner POSTs here so the browser gets real-time updates.
_SERVER_URL = os.environ.get("ASTRO_SERVER_URL", "http://127.0.0.1:8000")
_GEOMETRY_BATCH: list = []  # accumulate per tick, flush once


def _post_geometry_to_server(cell_id: str, geometry: dict) -> None:
    """Accumulate geometry for batch POST at end of tick."""
    _GEOMETRY_BATCH.append({"cell_id": cell_id, "geometry": geometry})


def _flush_geometry_batch() -> None:
    """POST all accumulated geometry updates to the server in one request."""
    global _GEOMETRY_BATCH
    if not _GEOMETRY_BATCH:
        return
    batch = _GEOMETRY_BATCH
    _GEOMETRY_BATCH = []
    try:
        payload = json.dumps({"cells": batch}).encode()
        req = urllib.request.Request(
            f"{_SERVER_URL}/api/cell/geometry",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass  # fire and forget
    except (urllib.error.URLError, OSError):
        pass  # server not running, skip silently

DEFAULT_ENERGY = 0.5
DEFAULT_DT_MS = 200


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def discover_cells():
    """Find every channels/cell/{id}/ directory that has a status.json."""
    cell_ids = []
    for name in sorted(os.listdir(CELL_DIR)):
        cell_path = os.path.join(CELL_DIR, name)
        if not os.path.isdir(cell_path):
            continue
        if os.path.isfile(os.path.join(cell_path, "status.json")):
            cell_ids.append(name)
    return cell_ids


# ─────────────────────────────────────────────────────────────────────────────
# Physics config (loaded once, shared by all cells/ticks)
# ─────────────────────────────────────────────────────────────────────────────

def load_physics():
    environment = load_json(
        os.path.join(PHYSICS_DIR, "environment.json"),
        default={
            "medium": {"temperature": 310},
            "flow_field": {"direction": [0.0, 0.0], "speed": 0.0, "turbulence": 0.0},
            "gravity": {"x": 0.0, "y": 0.0},
            "boundaries": {"width": 2048, "height": 2048, "margin": 50, "repel_force": 100},
            "brownian_noise": 0.0,
        },
    )
    lifecycle = load_json(
        os.path.join(PHYSICS_DIR, "cell_lifecycle.json"),
        default={
            "energy_system": {
                "base_consumption": 0.01,
                "movement_cost": 0.001,
                "collision_cost": 0.005,
                "regeneration_rate": 0.008,
                "max_energy": 1.0,
            },
            "lifecycle": {
                "division_energy_threshold": 0.9,
                "apoptosis_energy_threshold": 0.05,
            },
            "signaling": {"signal_radius": 200, "quorum_threshold": 4},
        },
    )
    interaction = load_json(
        os.path.join(PHYSICS_DIR, "species_interaction_matrix.json"),
        default={"interaction_radius": 300, "matrix": {}},
    )
    return environment, lifecycle, interaction


def interaction_strength(matrix, species_a, species_b):
    """G(a→b): attraction (positive) / repulsion (negative) coefficient.
    Unknown species pairs default to a mild neutral attraction (0.1) so
    the simulation degrades gracefully instead of crashing on new species."""
    row = matrix.get(species_a)
    if row is None:
        return 0.1
    return row.get(species_b, 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell state helpers
# ─────────────────────────────────────────────────────────────────────────────

def cell_center(status):
    """World-space center point for a cell, derived from status.position if
    present, else from its bbox."""
    pos = status.get("position")
    if pos and "x" in pos and "y" in pos:
        return float(pos["x"]), float(pos["y"])
    bbox = status.get("bbox", {})
    x = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2.0
    y = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2.0
    return x, y


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def deterministic_jitter(cell_id, tick, salt=""):
    """Small reproducible pseudo-random value in [-1, 1] per cell/tick, used
    for brownian-noise-style perturbation without needing a shared RNG seed
    across the whole sim."""
    rng = random.Random(f"{cell_id}:{tick}:{salt}")
    return rng.uniform(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tick logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_energy(status, lifecycle):
    """Step 4a — energy bookkeeping."""
    es = lifecycle.get("energy_system", {})
    base_consumption = es.get("base_consumption", 0.01)
    regeneration_rate = es.get("regeneration_rate", 0.008)
    max_energy = es.get("max_energy", 1.0)

    energy = status.get("energy")
    if energy is None:
        energy = DEFAULT_ENERGY

    energy = energy - base_consumption + regeneration_rate
    energy = clamp(energy, 0.0, max_energy)
    return energy


def compute_interaction_force(cell_id, species, center, all_statuses, interaction, lifecycle):
    """Step 4b — net attraction/repulsion vector from neighbors within
    signal_radius, weighted by the species_interaction_matrix and inverse
    distance. Returns (fx, fy, neighbor_count, nearest_same_species_id)."""
    matrix = interaction.get("matrix", {})
    signal_radius = lifecycle.get("signaling", {}).get("signal_radius", 200)

    fx, fy = 0.0, 0.0
    neighbor_count = 0
    nearest_same_species = None
    nearest_same_species_dist = float("inf")

    for other_id, other_status in all_statuses.items():
        if other_id == cell_id:
            continue
        other_center = cell_center(other_status)
        d = distance(center, other_center)
        if d <= 1e-6 or d > signal_radius:
            continue

        neighbor_count += 1
        other_species = other_status.get("species", "unknown")
        g = interaction_strength(matrix, species, other_species)

        # direction from self -> other, scaled by interaction strength and
        # inverse-distance falloff (closer neighbors push/pull harder).
        dx = (other_center[0] - center[0]) / d
        dy = (other_center[1] - center[1]) / d
        falloff = 1.0 - (d / signal_radius)
        fx += dx * g * falloff
        fy += dy * g * falloff

        if other_species == species and d < nearest_same_species_dist:
            nearest_same_species_dist = d
            nearest_same_species = other_id

    return fx, fy, neighbor_count, nearest_same_species


def compute_geometry(cell_id, tick, status, energy, lifecycle, environment,
                      neighbor_count, nearest_same_species, prev_geometry):
    """Step 4c/d/e — build a fresh geometry.json payload for this tick."""
    es = lifecycle.get("energy_system", {})
    max_energy = es.get("max_energy", 1.0)
    energy_norm = clamp(energy / max_energy, 0.0, 1.0)

    lc = lifecycle.get("lifecycle", {})
    division_threshold = lc.get("division_energy_threshold", 0.9)
    apoptosis_threshold = lc.get("apoptosis_energy_threshold", 0.05)

    quorum_threshold = lifecycle.get("signaling", {}).get("quorum_threshold", 4)
    quorum_active = neighbor_count >= quorum_threshold

    bbox = status.get("bbox", {})
    base_w = float(bbox.get("w", 40))
    base_h = float(bbox.get("h", 40))
    nominal_radius = max(4.0, min(base_w, base_h) / 2.5)

    # ── energy → size / surface smoothness ─────────────────────────────
    # high energy -> inflate + smooth, low energy -> shrink + rough
    size_factor = 0.8 + 0.4 * energy_norm  # 0.8x .. 1.2x
    base_radius = nominal_radius * size_factor

    noise_amplitude = clamp(0.09 - 0.08 * energy_norm, 0.005, 0.1)
    noise_frequency = clamp(2.0 + 4.0 * (1.0 - energy_norm), 1.0, 10.0)

    roughness = clamp(0.75 - 0.5 * energy_norm, 0.05, 0.9)
    glow_intensity = clamp(0.2 + 1.2 * energy_norm, 0.0, 2.0)
    opacity = clamp(0.4 + 0.55 * energy_norm, 0.0, 1.0)

    lobes = []
    last_action = "idle"
    division_ready = False

    if energy_norm > division_threshold:
        # ── division prep — peanut shape: two big lobes opposite each
        # other, base shrinks slightly to read as the "waist" pinch.
        division_ready = True
        last_action = "prepare_division"
        base_radius *= 0.85
        lobe_r = base_radius * 0.95
        lobes.append({"angle": 0.0, "distance": base_radius * 1.3, "radius": lobe_r})
        lobes.append({"angle": math.pi, "distance": base_radius * 1.3, "radius": lobe_r})

    elif energy_norm < apoptosis_threshold:
        # ── apoptosis — lobes scatter outward, fading opacity.
        last_action = "apoptosis"
        scatter_progress = clamp(1.0 - (energy_norm / max(apoptosis_threshold, 1e-6)), 0.0, 1.0)
        n_fragments = 5
        for i in range(n_fragments):
            angle = (2 * math.pi * i / n_fragments) + deterministic_jitter(cell_id, tick, f"frag{i}") * 0.3
            dist = base_radius * (1.0 + 2.5 * scatter_progress)
            radius = max(1.0, base_radius * 0.4 * (1.0 - 0.6 * scatter_progress))
            lobes.append({"angle": angle, "distance": dist, "radius": radius})
        opacity = clamp(opacity * (1.0 - scatter_progress), 0.0, 1.0)

    else:
        # ── normal range — 1-2 lobes, possibly a pseudopod toward the
        # nearest same-species neighbor (per SKILL.md: "近处有同 species →
        # 伸出伪足朝向邻居").
        n_lobes = 1 if energy_norm < 0.65 else 2
        for i in range(n_lobes):
            angle = (2 * math.pi * i / max(n_lobes, 1)) + deterministic_jitter(cell_id, tick, f"lobe{i}") * 0.5
            dist = base_radius * (0.6 + 0.3 * energy_norm)
            radius = base_radius * (0.5 + 0.3 * energy_norm)
            lobes.append({"angle": angle, "distance": dist, "radius": radius})
        last_action = "idle_metabolism"

    pseudopods = []
    if nearest_same_species is not None and energy_norm >= apoptosis_threshold:
        pseudopods.append({
            "target_cell": nearest_same_species,
            "length": clamp(base_radius * 1.8, 5.0, 60.0),
            "width": clamp(base_radius * 0.3, 1.0, 12.0),
            "curl": deterministic_jitter(cell_id, tick, "curl") * 0.4,
        })
        last_action = f"extend_pseudopod_toward_{nearest_same_species}"

    # ── flow field → stretch along flow direction ──────────────────────
    flow = environment.get("flow_field", {})
    flow_dir = flow.get("direction", [0.0, 0.0])
    flow_speed = flow.get("speed", 0.0)
    flow_angle = math.atan2(flow_dir[1], flow_dir[0]) if (flow_dir[0] or flow_dir[1]) else 0.0
    scale = clamp(1.0 + 0.002 * flow_speed, 0.85, 1.3)
    rotation = flow_angle * 0.1 + deterministic_jitter(cell_id, tick, "rot") * 0.02

    # ── temperature → membrane wobble frequency ─────────────────────────
    temperature = environment.get("medium", {}).get("temperature", 310)
    wobble_frequency = clamp((temperature - 280) / 20.0, 0.5, 8.0)
    wobble_amplitude = clamp(0.01 + 0.01 * (1.0 - energy_norm), 0.0, 0.05)

    # ── color: keep prior albedo/glow hue if present, else derive from
    # energy (cool/blue when low energy, warm/gold when high energy).
    prev_surface = (prev_geometry or {}).get("surface", {})
    if "albedo" in prev_surface:
        albedo = prev_surface["albedo"]
        glow_color = prev_surface.get("glow_color", [1.0, 0.8, 0.4])
    else:
        albedo = [
            round(0.3 + 0.5 * energy_norm, 3),
            round(0.4 + 0.3 * energy_norm, 3),
            round(0.7 - 0.3 * energy_norm, 3),
        ]
        glow_color = [1.0, 0.8, 0.4]

    if quorum_active:
        opacity = clamp(0.7 + 0.3 * math.sin(tick * 0.1), 0.0, 1.0)
        last_action = "quorum_sync"

    cx, cy = cell_center(status)

    geometry = {
        "cell_id": cell_id,
        "tick": tick,
        "timestamp_ms": int(time.time() * 1000),
        "transform": {
            "x": round(cx, 3),
            "y": round(cy, 3),
            "scale": round(scale, 4),
            "rotation": round(rotation, 4),
        },
        "sdf": {
            "type": "metaball",
            "base_radius": round(base_radius, 3),
            "lobes": [
                {
                    "angle": round(l["angle"], 4),
                    "distance": round(l["distance"], 3),
                    "radius": round(l["radius"], 3),
                }
                for l in lobes
            ],
            "noise_amplitude": round(noise_amplitude, 4),
            "noise_frequency": round(noise_frequency, 3),
        },
        "surface": {
            "albedo": albedo,
            "roughness": round(roughness, 3),
            "metallic": round(prev_surface.get("metallic", 0.1), 3),
            "opacity": round(opacity, 4),
            "glow_color": glow_color,
            "glow_intensity": round(glow_intensity, 3),
        },
        "membrane": {
            "thickness": round(2.0 + 1.0 * energy_norm, 3),
            "wobble_amplitude": round(wobble_amplitude, 4),
            "wobble_frequency": round(wobble_frequency, 3),
            "permeability_visual": round(lifecycle.get("membrane", {}).get("permeability", 0.5), 3),
        },
        "pseudopods": pseudopods,
        "internal_motion": {
            "cytoplasm_flow_angle": round(flow_angle, 4),
            "cytoplasm_flow_speed": round(clamp(flow_speed / 50.0, 0.0, 1.0), 4),
            "organelle_drift": round(0.01 + 0.02 * (1.0 - energy_norm), 4),
        },
    }

    return geometry, last_action, division_ready, quorum_active


def compute_position_update(status, fx, fy, environment, dt_s):
    """Apply interaction force + flow field + gravity + brownian noise to
    advance position by one tick. Soft-clamped to the environment boundary."""
    cx, cy = cell_center(status)
    vel = status.get("velocity", {"vx": 0.0, "vy": 0.0})
    vx, vy = float(vel.get("vx", 0.0)), float(vel.get("vy", 0.0))

    flow = environment.get("flow_field", {})
    flow_dir = flow.get("direction", [0.0, 0.0])
    flow_speed = flow.get("speed", 0.0)

    gravity = environment.get("gravity", {"x": 0.0, "y": 0.0})
    brownian = environment.get("brownian_noise", 0.0)

    accel_x = fx * 2.0 + flow_dir[0] * flow_speed * 0.02 + gravity.get("x", 0.0)
    accel_y = fy * 2.0 + flow_dir[1] * flow_speed * 0.02 + gravity.get("y", 0.0)

    accel_x += random.uniform(-1, 1) * brownian
    accel_y += random.uniform(-1, 1) * brownian

    vx = clamp(vx + accel_x * dt_s, -5.0, 5.0)
    vy = clamp(vy + accel_y * dt_s, -5.0, 5.0)

    new_x = cx + vx * dt_s
    new_y = cy + vy * dt_s

    boundaries = environment.get("boundaries", {})
    width = boundaries.get("width", 2048)
    height = boundaries.get("height", 2048)
    margin = boundaries.get("margin", 50)
    new_x = clamp(new_x, margin, max(margin, width - margin))
    new_y = clamp(new_y, margin, max(margin, height - margin))

    return new_x, new_y, vx, vy


# ─────────────────────────────────────────────────────────────────────────────
# Main tick loop
# ─────────────────────────────────────────────────────────────────────────────

def run_tick(cell_ids, environment, lifecycle, interaction, tick, dt_ms, verbose=False):
    dt_s = dt_ms / 1000.0

    # Load all statuses up front so neighbor sensing for this tick is
    # consistent (everyone reads the same world snapshot, per SKILL.md
    # step 1's "邻居感知 — 读取所有 cell 的 status.json").
    all_statuses = {}
    for cell_id in cell_ids:
        status_path = os.path.join(CELL_DIR, cell_id, "status.json")
        status = load_json(status_path, default={})
        status.setdefault("cell_id", cell_id)
        all_statuses[cell_id] = status

    for cell_id in cell_ids:
        status = all_statuses[cell_id]
        species = status.get("species", "unknown")
        center = cell_center(status)

        # 4a. energy
        energy = compute_energy(status, lifecycle)

        # 4b. interaction force
        fx, fy, neighbor_count, nearest_same_species = compute_interaction_force(
            cell_id, species, center, all_statuses, interaction, lifecycle
        )

        # geometry.json: previous value used to keep some continuity (color)
        geometry_path = os.path.join(CELL_DIR, cell_id, "geometry.json")
        prev_geometry = load_json(geometry_path, default=None)

        geometry, last_action, division_ready, quorum_active = compute_geometry(
            cell_id, tick, status, energy, lifecycle, environment,
            neighbor_count, nearest_same_species, prev_geometry,
        )

        # position update (movement has an energy cost)
        new_x, new_y, vx, vy = compute_position_update(status, fx, fy, environment, dt_s)
        if abs(vx) > 1e-3 or abs(vy) > 1e-3:
            es = lifecycle.get("energy_system", {})
            energy = clamp(energy - es.get("movement_cost", 0.001), 0.0, es.get("max_energy", 1.0))

        geometry["transform"]["x"] = round(new_x, 3)
        geometry["transform"]["y"] = round(new_y, 3)

        # 4f. write geometry.json (full overwrite)
        save_json(geometry_path, geometry)

        # 4f-sse. POST geometry to server for real-time SSE broadcast to browser
        _post_geometry_to_server(cell_id, geometry)

        # 4f. write status.json (full overwrite, preserving identity fields)
        new_status = dict(status)
        new_status["status"] = "alive" if energy >= lifecycle.get("lifecycle", {}).get(
            "apoptosis_energy_threshold", 0.05) else "apoptosing"
        new_status["epoch"] = tick
        new_status["energy"] = round(energy, 4)
        new_status["age_ms"] = status.get("age_ms", 0) + dt_ms
        new_status["position"] = {"x": round(new_x, 3), "y": round(new_y, 3)}
        new_status["velocity"] = {"vx": round(vx, 4), "vy": round(vy, 4)}
        new_status["quorum_active"] = quorum_active
        new_status["division_ready"] = division_ready
        new_status["last_action"] = last_action
        new_status.setdefault("community_id", status.get("community_id", 0))

        status_path = os.path.join(CELL_DIR, cell_id, "status.json")
        save_json(status_path, new_status)

        # 5. decision output out.json
        out_path = os.path.join(CELL_DIR, cell_id, "out.json")
        out_data = {
            "cell_id": cell_id,
            "tick": tick,
            "action": last_action,
            "target": nearest_same_species,
            "reason": (
                f"neighbors={neighbor_count} within signal_radius, "
                f"energy={round(energy, 3)}, species={species}"
            ),
            "energy_cost": lifecycle.get("energy_system", {}).get("base_consumption", 0.01),
            "geometry_changed": True,
        }
        save_json(out_path, out_data)

        # keep this cell's status visible to cells processed later in the
        # same tick (best-effort same-tick neighbor awareness)
        all_statuses[cell_id] = new_status

        if verbose:
            print(
                f"  [{cell_id:24s}] species={species:16s} energy={energy:.3f} "
                f"action={last_action} neighbors={neighbor_count}"
            )

    # Flush all geometry updates to server for SSE broadcast
    _flush_geometry_batch()


def main():
    parser = argparse.ArgumentParser(description="Run the cell-pubsub tick loop for N ticks.")
    parser.add_argument("--ticks", type=int, default=1, help="number of ticks to simulate")
    parser.add_argument("--dt-ms", type=int, default=DEFAULT_DT_MS, help="simulated ms per tick")
    parser.add_argument("--seed", type=int, default=None, help="random seed for brownian motion")
    parser.add_argument("--verbose", action="store_true", help="print per-cell per-tick summary")
    parser.add_argument("--live", action="store_true",
                        help="run forever at --interval-ms pace, POSTing geometry to server")
    parser.add_argument("--interval-ms", type=int, default=500,
                        help="sleep between ticks in --live mode (ms)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cell_ids = discover_cells()
    if not cell_ids:
        print(f"No cells found under {CELL_DIR}", file=sys.stderr)
        sys.exit(1)

    environment, lifecycle, interaction = load_physics()

    print(f"tick-runner: {len(cell_ids)} cells, {args.ticks} tick(s), dt={args.dt_ms}ms")

    start_epochs = {}
    for cell_id in cell_ids:
        status = load_json(os.path.join(CELL_DIR, cell_id, "status.json"), default={})
        start_epochs[cell_id] = status.get("epoch", 0)

    base_epoch = max(start_epochs.values()) if start_epochs else 0

    if args.live:
        # Live mode: run ticks forever, POSTing geometry to server each tick
        print(f"tick-runner: LIVE MODE — {len(cell_ids)} cells, interval={args.interval_ms}ms")
        print(f"  server: {_SERVER_URL}")
        tick = base_epoch
        try:
            while True:
                tick += 1
                run_tick(cell_ids, environment, lifecycle, interaction,
                         tick, args.dt_ms, verbose=args.verbose)
                if tick % 10 == 0:
                    print(f"tick-runner: tick {tick} done")
                time.sleep(args.interval_ms / 1000.0)
        except KeyboardInterrupt:
            print(f"\ntick-runner: stopped at tick {tick}")
        return

    for i in range(args.ticks):
        tick = base_epoch + i + 1
        if args.verbose:
            print(f"--- tick {tick} ---")
        run_tick(cell_ids, environment, lifecycle, interaction, tick, args.dt_ms, verbose=args.verbose)

    print(f"tick-runner: done. {len(cell_ids)} cells advanced to epoch {base_epoch + args.ticks}.")


if __name__ == "__main__":
    main()
