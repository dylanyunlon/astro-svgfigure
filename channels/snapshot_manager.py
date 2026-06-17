#!/usr/bin/env python3
"""
snapshot_manager.py — M174: params-level divergence rollback snapshot.

Saves cell/*/params.json at the end of every epoch into
  channels/convergence/epoch_params/{epoch}/{cell_id}.json

On divergence (params delta spike > DIVERGENCE_FACTOR × previous max_delta)
restores the entire epoch's params from the most-recent clean snapshot and
returns the rollback target epoch so the caller can re-drive the loop.

Ported from FAstroParamSnapshotManager (upstream/unreal-renderer/
SceneCaptureRendering.cpp commit d31c85e) — same ring-buffer capacity
(MAX_HISTORY = 16) and rollback semantics as EpochSnapshotManager, but
tracking the full params.json payload rather than bbox-only state.

Mapping (C++ → Python):
  FAstroParamSnapshot          → ParamSnapshot  (dict, JSON-serialisable)
  CaptureParamSnapshot()       → ParamsSnapshotManager.capture(epoch)
  RollbackToParamSnapshot()    → ParamsSnapshotManager.rollback(epoch)
  CheckParamDivergence()       → ParamsSnapshotManager.check_divergence(max_delta, prev_max_delta)
  FAstroParamDelta::Norm()     → _l2_delta()

Public API
──────────
  mgr = ParamsSnapshotManager(channels_root)
  mgr.capture(epoch)                 → snapshot dict (also written to disk)
  mgr.rollback(target_epoch)         → bool  (True on success)
  mgr.check_divergence(cur, prev)    → bool  (True → diverged + rollback triggered)
  mgr.latest_snapshot_epoch          → int | None
  mgr.history_count                  → int
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import time as _time
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Ring-buffer capacity — mirrors kMaxSnapshotHistory = 16 in C++ source.
# ---------------------------------------------------------------------------
_MAX_HISTORY = 16

# Divergence factor: if current max_delta > FACTOR × previous max_delta we
# treat the epoch as diverging and roll back.  Matches the default used by
# convergence_check() in loop_orchestrator.py (M173).
_DIVERGENCE_FACTOR_DEFAULT = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_params(params: dict) -> List[float]:
    """
    Extract numeric scalars from a params dict into a flat vector.

    Fields included (mirrors FAstroParamDelta field list from d31c85e):
      bbox         : x, y, w, h, z
      top-level    : opacity, font_size, z
      species_params: every numeric value (species-agnostic; sorted for determinism)
      shadow       : dx, dy, blur, opacity
      bloom        : threshold, bloomScale, strength, blur, tint[R,G,B]

    Non-numeric fields (strings, bools) are silently skipped.
    """
    vec: List[float] = []

    # bbox
    bbox = params.get("bbox", {})
    for k in ("x", "y", "w", "h", "z"):
        v = bbox.get(k)
        if isinstance(v, (int, float)):
            vec.append(float(v))

    # top-level numeric scalars
    for k in ("opacity", "font_size", "z"):
        v = params.get(k)
        if isinstance(v, (int, float)):
            vec.append(float(v))

    # species_params — all numeric values, sorted for determinism
    sp = params.get("species_params", {})
    if isinstance(sp, dict):
        for k in sorted(sp):
            v = sp[k]
            if isinstance(v, (int, float)):
                vec.append(float(v))

    # shadow
    shadow = params.get("shadow", {})
    if isinstance(shadow, dict):
        for k in ("dx", "dy", "blur", "opacity"):
            v = shadow.get(k)
            if isinstance(v, (int, float)):
                vec.append(float(v))

    # bloom (M027 AdvancedBloomFilter — forward-compatible; tracked when present)
    bloom = params.get("bloom", {})
    if isinstance(bloom, dict):
        for k in ("threshold", "bloomScale", "strength", "blur"):
            v = bloom.get(k)
            if isinstance(v, (int, float)):
                vec.append(float(v))
        # tint is a [R, G, B] list
        tint = bloom.get("tint")
        if isinstance(tint, (list, tuple)):
            for ch in tint:
                if isinstance(ch, (int, float)):
                    vec.append(float(ch))

    return vec


def _l2_delta(vec_a: List[float], vec_b: List[float]) -> float:
    """
    L2 norm of (vec_a − vec_b).  Pads shorter vector with zeros.
    Mirrors FAstroParamDelta::Norm().
    """
    n = max(len(vec_a), len(vec_b))
    total = 0.0
    for i in range(n):
        va = vec_a[i] if i < len(vec_a) else 0.0
        vb = vec_b[i] if i < len(vec_b) else 0.0
        d = va - vb
        total += d * d
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# ParamsSnapshotManager
# ---------------------------------------------------------------------------

class ParamsSnapshotManager:
    """
    Params-level epoch snapshot manager.

    Saves the full params.json payload for every cell at each epoch so that:
      1. convergence_check() (M173) can compare adjacent epoch params without
         re-reading the file pair itself — it delegates snapshot I/O here.
      2. On divergence the entire epoch's params can be restored in one call.

    Snapshot layout on disk:
        channels/convergence/epoch_params/{epoch}/{cell_id}.json

    In-memory ring buffer (capacity MAX_HISTORY = 16):
        _history  list of ParamSnapshot dicts, oldest→newest
    """

    MAX_HISTORY: int = _MAX_HISTORY

    def __init__(self, channels_root: str) -> None:
        self._root = channels_root
        self._history: List[dict] = []      # ring buffer
        self._epoch_params_dir = os.path.join(
            channels_root, "convergence", "epoch_params"
        )
        os.makedirs(self._epoch_params_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public: capture
    # ------------------------------------------------------------------
    def capture(self, epoch: int) -> dict:
        """
        Read all cell/*/params.json for the given *epoch* index, persist each
        to convergence/epoch_params/{epoch}/{cell_id}.json, and push a summary
        snapshot into the in-memory ring buffer.

        Returns the snapshot dict:
          {
            "epoch":     <int>,
            "ts":        <float ms>,
            "cell_ids":  [<str>, ...],
            "cell_params": { "<cell_id>": <params_dict>, ... },
            "max_vec_norm": <float>   # max ‖params_vector‖ across cells
          }

        Mirrors CaptureParamSnapshot() including the progress log line.
        """
        cell_params = self._read_all_params()
        epoch_dir = os.path.join(self._epoch_params_dir, str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)

        # Persist individual cell snapshots to disk
        for cell_id, params in cell_params.items():
            snap_path = os.path.join(epoch_dir, f"{cell_id}.json")
            try:
                with open(snap_path, "w") as f:
                    json.dump(params, f, indent=2)
            except OSError as exc:
                print(
                    f"[M174] WARNING: could not write {snap_path}: {exc}",
                    file=sys.stderr,
                )

        # Compute max L2 norm of each cell's params vector (used for divergence
        # trending — a rising norm indicates the system is drifting).
        max_vec_norm = 0.0
        for params in cell_params.values():
            norm = math.sqrt(sum(v * v for v in _flatten_params(params)))
            if norm > max_vec_norm:
                max_vec_norm = norm

        snap: dict = {
            "epoch":         epoch,
            "ts":            _time.time() * 1000.0,
            "cell_ids":      sorted(cell_params),
            "cell_params":   cell_params,
            "max_vec_norm":  max_vec_norm,
        }

        # Ring buffer: drop oldest when full
        self._history.append(snap)
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)

        print(
            f"[M174] capture: epoch={epoch} "
            f"cells={len(cell_params)} "
            f"max_vec_norm={max_vec_norm:.4f} "
            f"ts={snap['ts']:.0f}"
        )
        return snap

    # ------------------------------------------------------------------
    # Public: rollback
    # ------------------------------------------------------------------
    def rollback(self, target_epoch: int) -> bool:
        """
        Restore cell/*/params.json from the snapshot saved at *target_epoch*.

        Reads from convergence/epoch_params/{target_epoch}/*.json and writes
        back to cell/{cell_id}/params.json.  Also trims the in-memory ring
        buffer to exclude epochs after the rollback target (mirrors C++
        ``_history = [s for s in _history if s.epoch <= target_epoch]``).

        Returns True on success, False when target snapshot is not found.
        Mirrors RollbackToParamSnapshot().
        """
        # First try in-memory ring buffer (fast path)
        snap = self._find_snapshot(target_epoch)

        if snap is not None:
            cell_params = snap["cell_params"]
        else:
            # Fall back to disk — load from convergence/epoch_params/{epoch}/
            epoch_dir = os.path.join(self._epoch_params_dir, str(target_epoch))
            if not os.path.isdir(epoch_dir):
                print(
                    f"[M174] rollback: epoch {target_epoch} not in history "
                    f"(history_count={len(self._history)}, "
                    f"disk_dir={epoch_dir} missing)",
                    file=sys.stderr,
                )
                return False
            cell_params = {}
            for f_name in os.listdir(epoch_dir):
                if not f_name.endswith(".json"):
                    continue
                cell_id = f_name[:-5]  # strip .json
                try:
                    with open(os.path.join(epoch_dir, f_name)) as f:
                        cell_params[cell_id] = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    print(
                        f"[M174] rollback: could not read {f_name}: {exc}",
                        file=sys.stderr,
                    )

        if not cell_params:
            print(
                f"[M174] rollback: epoch {target_epoch} snapshot is empty",
                file=sys.stderr,
            )
            return False

        # Write restored params back to cell/{cell_id}/params.json
        cell_dir = os.path.join(self._root, "cell")
        restored = 0
        for cell_id, params in cell_params.items():
            params_path = os.path.join(cell_dir, cell_id, "params.json")
            cell_subdir = os.path.dirname(params_path)
            if not os.path.isdir(cell_subdir):
                continue
            try:
                with open(params_path, "w") as f:
                    json.dump(params, f, indent=2)
                restored += 1
            except OSError as exc:
                print(
                    f"[M174] rollback: could not restore {cell_id}: {exc}",
                    file=sys.stderr,
                )

        # Trim ring buffer — drop snapshots newer than target
        self._history = [s for s in self._history if s["epoch"] <= target_epoch]

        print(
            f"[M174] rollback: restored epoch={target_epoch} "
            f"cells={restored}"
        )
        return restored > 0

    # ------------------------------------------------------------------
    # Public: check_divergence
    # ------------------------------------------------------------------
    def check_divergence(
        self,
        cur_max_delta: float,
        prev_max_delta: float,
        epoch: int,
        divergence_factor: float = _DIVERGENCE_FACTOR_DEFAULT,
    ) -> bool:
        """
        Detect a sudden param-delta spike and trigger rollback.

        Returns True (diverged) when:
          prev_max_delta > 1e-6  AND  cur_max_delta > divergence_factor × prev_max_delta

        On divergence, automatically calls rollback(epoch - 1).
        Mirrors CheckParamDivergence() / FAstroConvergenceController.

        Args:
            cur_max_delta    : max L2 delta norm across cells this epoch.
            prev_max_delta   : max L2 delta norm from the previous epoch.
            epoch            : current epoch index (0-based).
            divergence_factor: multiplier threshold (default 3.0).
        """
        if prev_max_delta > 1e-6 and cur_max_delta > divergence_factor * prev_max_delta:
            print(
                f"[M174] DIVERGENCE: epoch={epoch} "
                f"max_delta={cur_max_delta:.4f} > "
                f"{divergence_factor}× prev={prev_max_delta:.4f} "
                f"— rolling back to epoch {epoch - 1}"
            )
            self.rollback(epoch - 1)
            return True

        print(
            f"[M174] params OK: epoch={epoch} "
            f"delta {prev_max_delta:.4f}→{cur_max_delta:.4f}"
        )
        return False

    # ------------------------------------------------------------------
    # Public: load_epoch_params
    # ------------------------------------------------------------------
    def load_epoch_params(self, epoch: int) -> Dict[str, dict]:
        """
        Load per-cell params for *epoch* from disk (convergence/epoch_params/).
        Returns a dict keyed by cell_id, or {} if the snapshot does not exist.

        Used by convergence_check() to read the previous epoch's params without
        duplicating the glob logic.
        """
        epoch_dir = os.path.join(self._epoch_params_dir, str(epoch))
        if not os.path.isdir(epoch_dir):
            return {}
        result: Dict[str, dict] = {}
        for fname in sorted(os.listdir(epoch_dir)):
            if not fname.endswith(".json"):
                continue
            cell_id = fname[:-5]
            try:
                with open(os.path.join(epoch_dir, fname)) as f:
                    result[cell_id] = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return result

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def latest_snapshot_epoch(self) -> Optional[int]:
        """Epoch index of the most-recent in-memory snapshot, or None."""
        return self._history[-1]["epoch"] if self._history else None

    @property
    def history_count(self) -> int:
        """Number of snapshots currently held in the ring buffer."""
        return len(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _read_all_params(self) -> Dict[str, dict]:
        """Read cell/*/params.json → {cell_id: params_dict}."""
        pattern = os.path.join(self._root, "cell", "*", "params.json")
        result: Dict[str, dict] = {}
        for path in sorted(glob.glob(pattern)):
            cell_id = os.path.basename(os.path.dirname(path))
            try:
                with open(path) as f:
                    result[cell_id] = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(
                    f"[M174] WARNING: could not read {path}: {exc}",
                    file=sys.stderr,
                )
        return result

    def _find_snapshot(self, epoch_idx: int) -> Optional[dict]:
        """Linear scan newest→oldest (O(N), N ≤ MAX_HISTORY=16)."""
        for snap in reversed(self._history):
            if snap["epoch"] == epoch_idx:
                return snap
        return None


# ---------------------------------------------------------------------------
# Convenience function: compute per-cell delta norms between two epochs
# ---------------------------------------------------------------------------

def compute_param_deltas(
    prev_params: Dict[str, dict],
    curr_params: Dict[str, dict],
) -> Dict[str, float]:
    """
    Compute per-cell L2 delta norm of the flattened params vector
    between two consecutive epoch snapshots.

    Returns {cell_id: delta_norm} for cells present in *curr_params*.
    Cells absent from *prev_params* get delta 0.0 (first appearance).

    Used by convergence_check() (M173) to get per-cell deltas without
    inlining the flatten/l2 arithmetic.
    """
    deltas: Dict[str, float] = {}
    for cell_id, curr in curr_params.items():
        vec_curr = _flatten_params(curr)
        prev = prev_params.get(cell_id)
        if prev is None:
            deltas[cell_id] = 0.0
        else:
            deltas[cell_id] = round(_l2_delta(vec_curr, _flatten_params(prev)), 6)
    return deltas


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description="M174 snapshot_manager — CLI smoke-test"
    )
    _parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="channels/ root (default: directory containing this script)",
    )
    _parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch index to capture (default: 0)",
    )
    _parser.add_argument(
        "--rollback",
        type=int,
        default=None,
        metavar="EPOCH",
        help="Roll back to EPOCH and exit",
    )
    _args = _parser.parse_args()

    mgr = ParamsSnapshotManager(_args.root)

    if _args.rollback is not None:
        ok = mgr.rollback(_args.rollback)
        sys.exit(0 if ok else 1)

    snap = mgr.capture(_args.epoch)
    print(
        f"Snapshot saved: epoch={snap['epoch']} "
        f"cells={len(snap['cell_ids'])} "
        f"max_vec_norm={snap['max_vec_norm']:.4f}"
    )
