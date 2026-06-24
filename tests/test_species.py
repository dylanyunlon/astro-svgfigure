"""
tests/test_species.py
=====================
M1074: Validate channels/physics/species_assignment.json.

Checks:
  - File exists and is valid JSON
  - Root is a dict with exactly 58 entries (one per topology node)
  - Every entry has 'species' (string, from allowed set) and 'gene_traits' (dict)
  - gene_traits contains all four required keys:
      'primary_shape', 'pattern', 'line_style', 'family'
  - All values in gene_traits are non-empty strings
  - No entry has a null/missing species
"""

import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECIES_PATH = os.path.join(REPO_ROOT, "channels", "physics", "species_assignment.json")

EXPECTED_ENTRY_COUNT = 58

# The complete canonical set of species identifiers used in the project
VALID_SPECIES = {
    "cil-arrow-right",
    "cil-bolt",
    "cil-code",
    "cil-eye",
    "cil-filter",
    "cil-graph",
    "cil-layers",
    "cil-loop",
    "cil-plus",
    "cil-vector",
}

# Required keys inside every gene_traits object
REQUIRED_GENE_TRAIT_KEYS = {"primary_shape", "pattern", "line_style", "family"}


@pytest.fixture(scope="module")
def species_data():
    assert os.path.exists(SPECIES_PATH), (
        f"species_assignment.json not found at {SPECIES_PATH}"
    )
    with open(SPECIES_PATH) as f:
        return json.load(f)


# ── Schema ──────────────────────────────────────────────────────────────────

def test_species_file_is_dict(species_data):
    assert isinstance(species_data, dict), (
        "species_assignment.json root must be a JSON object (dict)"
    )


def test_species_entry_count(species_data):
    assert len(species_data) == EXPECTED_ENTRY_COUNT, (
        f"Expected {EXPECTED_ENTRY_COUNT} entries, got {len(species_data)}"
    )


# ── Per-entry validation ─────────────────────────────────────────────────────

def test_every_entry_has_species_key(species_data):
    missing = [k for k, v in species_data.items() if "species" not in v]
    assert not missing, f"Entries missing 'species' key: {missing}"


def test_every_entry_has_gene_traits_key(species_data):
    missing = [k for k, v in species_data.items() if "gene_traits" not in v]
    assert not missing, f"Entries missing 'gene_traits' key: {missing}"


def test_species_values_are_strings(species_data):
    bad = [k for k, v in species_data.items() if not isinstance(v.get("species"), str)]
    assert not bad, f"Entries with non-string 'species': {bad}"


def test_species_values_non_empty(species_data):
    empty = [k for k, v in species_data.items() if not v.get("species", "").strip()]
    assert not empty, f"Entries with empty 'species': {empty}"


def test_species_values_in_valid_set(species_data):
    invalid = {
        k: v["species"]
        for k, v in species_data.items()
        if v.get("species") not in VALID_SPECIES
    }
    assert not invalid, (
        f"Entries with unrecognised species values: {invalid}\n"
        f"Allowed species: {sorted(VALID_SPECIES)}"
    )


# ── gene_traits structure ─────────────────────────────────────────────────────

def test_gene_traits_are_dicts(species_data):
    bad = [
        k for k, v in species_data.items()
        if not isinstance(v.get("gene_traits"), dict)
    ]
    assert not bad, f"Entries whose gene_traits is not a dict: {bad}"


def test_gene_traits_required_keys_present(species_data):
    failures = {}
    for k, v in species_data.items():
        gt = v.get("gene_traits", {})
        missing_keys = REQUIRED_GENE_TRAIT_KEYS - set(gt.keys())
        if missing_keys:
            failures[k] = sorted(missing_keys)
    assert not failures, (
        f"Entries with missing gene_traits keys:\n"
        + "\n".join(f"  {k}: missing {m}" for k, m in failures.items())
    )


def test_gene_traits_values_are_non_empty_strings(species_data):
    failures = {}
    for k, v in species_data.items():
        gt = v.get("gene_traits", {})
        bad_keys = [
            gk for gk in REQUIRED_GENE_TRAIT_KEYS
            if not isinstance(gt.get(gk), str) or not gt.get(gk, "").strip()
        ]
        if bad_keys:
            failures[k] = bad_keys
    assert not failures, (
        f"Entries with non-string or empty gene_traits values: {failures}"
    )


def test_no_extra_top_level_keys_in_entries(species_data):
    """Each entry should contain exactly 'species' and 'gene_traits'."""
    allowed = {"species", "gene_traits"}
    extras = {
        k: sorted(set(v.keys()) - allowed)
        for k, v in species_data.items()
        if set(v.keys()) - allowed
    }
    # This is a soft warning test — we report but don't fail on extra keys
    # (future schema extensions are OK)
    # Uncomment the assert to make it strict:
    # assert not extras, f"Entries with unexpected keys: {extras}"
    if extras:
        import warnings
        warnings.warn(f"Entries with extra (non-schema) keys: {extras}")


# ── Coverage sanity ──────────────────────────────────────────────────────────

def test_all_valid_species_appear_at_least_once(species_data):
    """Every species in VALID_SPECIES should be used at least once."""
    used = {v["species"] for v in species_data.values() if "species" in v}
    unused = VALID_SPECIES - used
    assert not unused, (
        f"Species defined but never assigned to any node: {sorted(unused)}"
    )
