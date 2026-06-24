"""
tests/test_convergence.py
=========================
M1074: Validate channels/convergence/status.json format.

Checks:
  - File exists and is valid JSON
  - Root is a dict
  - Required core keys present: epoch, converged, max_delta, cell_deltas,
    diverged, threshold
  - Optional enrichment keys (source_epoch, verified_by, note) validated
    when present
  - Type correctness for each field
  - Invariant: converged XOR diverged (cannot both be True)
  - max_delta is non-negative float
  - threshold is positive float
  - cell_deltas is a dict mapping cell names → non-negative floats
  - If converged == True then max_delta <= threshold
"""

import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_PATH = os.path.join(REPO_ROOT, "channels", "convergence", "status.json")

# Core keys always present in every emission of status.json
CORE_KEYS = {
    "epoch",
    "converged",
    "max_delta",
    "cell_deltas",
    "diverged",
    "threshold",
}

# Optional enrichment keys written by verified/annotated epochs
OPTIONAL_KEYS = {"source_epoch", "verified_by", "note"}


@pytest.fixture(scope="module")
def status():
    assert os.path.exists(STATUS_PATH), (
        f"convergence/status.json not found at {STATUS_PATH}"
    )
    with open(STATUS_PATH) as f:
        return json.load(f)


# ── Schema ──────────────────────────────────────────────────────────────────

def test_status_is_dict(status):
    assert isinstance(status, dict), "status.json root must be a JSON object"


def test_core_keys_present(status):
    missing = CORE_KEYS - set(status.keys())
    assert not missing, f"status.json is missing required keys: {sorted(missing)}"


def test_no_unknown_keys(status):
    """Every key in status.json must be in CORE_KEYS ∪ OPTIONAL_KEYS."""
    known = CORE_KEYS | OPTIONAL_KEYS
    unknown = set(status.keys()) - known
    assert not unknown, (
        f"status.json contains unexpected keys: {sorted(unknown)}\n"
        f"Known keys: {sorted(known)}"
    )


# ── Type checks — core ───────────────────────────────────────────────────────

def test_epoch_is_int(status):
    assert isinstance(status["epoch"], int), (
        f"'epoch' must be int, got {type(status['epoch']).__name__}"
    )


def test_epoch_is_non_negative(status):
    assert status["epoch"] >= 0, f"'epoch' must be >= 0, got {status['epoch']}"


def test_converged_is_bool(status):
    assert isinstance(status["converged"], bool), (
        f"'converged' must be bool, got {type(status['converged']).__name__}"
    )


def test_diverged_is_bool(status):
    assert isinstance(status["diverged"], bool), (
        f"'diverged' must be bool, got {type(status['diverged']).__name__}"
    )


def test_max_delta_is_number(status):
    assert isinstance(status["max_delta"], (int, float)), (
        f"'max_delta' must be numeric, got {type(status['max_delta']).__name__}"
    )


def test_max_delta_non_negative(status):
    assert status["max_delta"] >= 0, (
        f"'max_delta' must be >= 0, got {status['max_delta']}"
    )


def test_threshold_is_positive_number(status):
    val = status["threshold"]
    assert isinstance(val, (int, float)), (
        f"'threshold' must be numeric, got {type(val).__name__}"
    )
    assert val > 0, f"'threshold' must be > 0, got {val}"


# ── Type checks — optional enrichment keys ───────────────────────────────────

def test_source_epoch_is_int_when_present(status):
    if "source_epoch" not in status:
        pytest.skip("source_epoch not present in this epoch's status.json")
    val = status["source_epoch"]
    assert isinstance(val, int), (
        f"'source_epoch' must be int, got {type(val).__name__}"
    )


def test_verified_by_is_non_empty_string_when_present(status):
    if "verified_by" not in status:
        pytest.skip("verified_by not present in this epoch's status.json")
    vb = status["verified_by"]
    assert isinstance(vb, str) and vb.strip(), (
        f"'verified_by' must be a non-empty string, got {vb!r}"
    )


def test_note_is_non_empty_string_when_present(status):
    if "note" not in status:
        pytest.skip("note not present in this epoch's status.json")
    note = status["note"]
    assert isinstance(note, str) and note.strip(), (
        f"'note' must be a non-empty string, got {note!r}"
    )


def test_source_epoch_gte_epoch_when_present(status):
    if "source_epoch" not in status:
        pytest.skip("source_epoch not present in this epoch's status.json")
    assert status["source_epoch"] >= status["epoch"], (
        f"source_epoch={status['source_epoch']} < epoch={status['epoch']}"
    )


# ── cell_deltas ──────────────────────────────────────────────────────────────

def test_cell_deltas_is_dict(status):
    assert isinstance(status["cell_deltas"], dict), (
        f"'cell_deltas' must be a dict, got {type(status['cell_deltas']).__name__}"
    )


def test_cell_deltas_non_empty(status):
    assert len(status["cell_deltas"]) > 0, "'cell_deltas' must not be empty"


def test_cell_deltas_values_are_non_negative_numbers(status):
    bad = {
        k: v for k, v in status["cell_deltas"].items()
        if not isinstance(v, (int, float)) or v < 0
    }
    assert not bad, (
        f"cell_deltas entries must be non-negative numbers; bad entries: {bad}"
    )


def test_cell_deltas_keys_are_strings(status):
    bad = [k for k in status["cell_deltas"] if not isinstance(k, str)]
    assert not bad, f"cell_deltas keys must be strings; bad: {bad}"


# ── Logical invariants ───────────────────────────────────────────────────────

def test_converged_and_diverged_mutually_exclusive(status):
    assert not (status["converged"] and status["diverged"]), (
        "'converged' and 'diverged' cannot both be True simultaneously"
    )


def test_converged_implies_max_delta_within_threshold(status):
    """When converged=True, max_delta must be <= threshold."""
    if status["converged"]:
        assert status["max_delta"] <= status["threshold"], (
            f"converged=True but max_delta={status['max_delta']} "
            f"> threshold={status['threshold']}"
        )


def test_max_delta_matches_cell_deltas_max(status):
    """max_delta must equal the maximum value in cell_deltas."""
    if not status["cell_deltas"]:
        pytest.skip("cell_deltas is empty")
    expected_max = max(status["cell_deltas"].values())
    assert abs(status["max_delta"] - expected_max) < 1e-9, (
        f"max_delta={status['max_delta']} does not match "
        f"max(cell_deltas)={expected_max}"
    )
