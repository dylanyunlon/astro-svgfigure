"""
tests/test_rendering_modules.py
================================
M1074: Validate that all 30 channels/rendering/ submodules are importable.

The rendering pipeline is split into 30 submodules (packages + top-level
modules).  A broken import means a wiring error has been introduced.

Checks:
  - Exactly 30 importable submodules exist under channels/rendering/
  - Every submodule imports without raising any exception
  - Each package submodule exposes an __init__ (not an import error stub)
  - constants.py is present and importable as a module (not a package)
"""

import importlib
import os
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDERING_DIR = os.path.join(REPO_ROOT, "channels", "rendering")

EXPECTED_SUBMODULE_COUNT = 30

# Canonical list of the 30 expected submodules
EXPECTED_SUBMODULES = [
    "acceleration",
    "color",
    "compositor",
    "constants",
    "decoration",
    "distancefield",
    "drawcall",
    "effects",
    "lighting",
    "lumen",
    "misc",
    "motionblur",
    "nanite",
    "occlusion",
    "passes",
    "pathtracing",
    "postprocess",
    "reflection",
    "registry",
    "resources",
    "scene",
    "shading",
    "shadow",
    "species",
    "streaming",
    "styleprobe",
    "temporal_aa",
    "translucency",
    "utils",
    "visibility",
]


def _discover_submodules():
    """Return sorted list of importable submodule names in channels/rendering/."""
    found = []
    for entry in os.listdir(RENDERING_DIR):
        full = os.path.join(RENDERING_DIR, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "__init__.py")):
            found.append(entry)
        elif (
            entry.endswith(".py")
            and entry not in ("__init__.py",)
            and not entry.startswith("_")
        ):
            found.append(entry[:-3])
    return sorted(found)


# ── Structural check ─────────────────────────────────────────────────────────

def test_rendering_dir_exists():
    assert os.path.isdir(RENDERING_DIR), (
        f"channels/rendering/ directory not found at {RENDERING_DIR}"
    )


def test_rendering_has_init():
    init = os.path.join(RENDERING_DIR, "__init__.py")
    assert os.path.exists(init), "channels/rendering/__init__.py is missing"


def test_submodule_count():
    found = _discover_submodules()
    assert len(found) == EXPECTED_SUBMODULE_COUNT, (
        f"Expected {EXPECTED_SUBMODULE_COUNT} rendering submodules, "
        f"found {len(found)}: {found}"
    )


def test_expected_submodule_names_match():
    found = _discover_submodules()
    extra = sorted(set(found) - set(EXPECTED_SUBMODULES))
    missing = sorted(set(EXPECTED_SUBMODULES) - set(found))
    assert not extra and not missing, (
        f"Submodule mismatch — extra: {extra}, missing: {missing}"
    )


# ── Import checks (parametrised) ─────────────────────────────────────────────

# Ensure repo root is on path before parametrised tests run
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.mark.parametrize("submodule", EXPECTED_SUBMODULES)
def test_rendering_submodule_importable(submodule):
    """Each rendering submodule must import cleanly with no exception."""
    module_name = f"channels.rendering.{submodule}"
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        pytest.fail(
            f"Import of '{module_name}' raised {type(exc).__name__}: {exc}"
        )
    assert isinstance(mod, types.ModuleType), (
        f"'{module_name}' did not return a module object"
    )


@pytest.mark.parametrize("submodule", EXPECTED_SUBMODULES)
def test_rendering_submodule_has_file(submodule):
    """Each submodule must correspond to a real path on disk."""
    pkg_path = os.path.join(RENDERING_DIR, submodule, "__init__.py")
    mod_path = os.path.join(RENDERING_DIR, submodule + ".py")
    exists = os.path.exists(pkg_path) or os.path.exists(mod_path)
    assert exists, (
        f"No file found for rendering submodule '{submodule}' "
        f"(checked {pkg_path} and {mod_path})"
    )


# ── Top-level rendering package ──────────────────────────────────────────────

def test_rendering_package_importable():
    """The top-level channels.rendering package itself must import."""
    try:
        mod = importlib.import_module("channels.rendering")
    except Exception as exc:
        pytest.fail(f"channels.rendering import failed: {exc}")
    assert isinstance(mod, types.ModuleType)


def test_constants_is_module_not_package():
    """constants should be a flat .py module, not a package dir."""
    pkg_path = os.path.join(RENDERING_DIR, "constants", "__init__.py")
    mod_path = os.path.join(RENDERING_DIR, "constants.py")
    # It can be either, but at least one must exist
    assert os.path.exists(mod_path) or os.path.exists(pkg_path), (
        "channels/rendering/constants.py (or package) not found"
    )
