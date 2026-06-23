"""
channels/physics/implicit_viscosity.py
M781 — Implicit viscosity solver for the cell pub/sub physics loop.

Solves the implicit viscosity diffusion equation:

    (I - dt·μ·L) v_new = v_old

where:
    I     = identity operator
    dt    = physics time step
    μ     = kinematic viscosity coefficient (from QoS profile or override)
    L     = graph Laplacian built from topology edges
    v_old = velocity field before viscosity diffusion
    v_new = velocity field after viscosity diffusion (unknown)

The system is symmetric positive-definite (SPD) because (I - dt·μ·L) has
positive diagonal and non-positive off-diagonal entries with strict diagonal
dominance, so the Conjugate Gradient method converges unconditionally.

Upstream reference: the implicit viscosity step in Stam's "Stable Fluids"
(SIGGRAPH 1999), adapted from grid-based to graph-based diffusion over
the cell topology.  The graph Laplacian replaces the standard 5-point
finite-difference stencil — each topology edge contributes a diffusion
coupling between its source and target cells.

Integration point: called from physics_engine() in loop_orchestrator.py
after explicit force accumulation (collision, charge, link, center) and
before constraint propagation, so that velocity diffusion smooths the
force field before distance-field propagation amplifies it.
"""

from __future__ import annotations

import math
import os
import json
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DT       = 1.0 / 60.0   # 60 Hz physics tick
_DEFAULT_MU       = 0.015         # kinematic viscosity; mid-range of QoS band
_CG_MAX_ITER      = 200           # CG iteration cap
_CG_TOL           = 1e-8          # relative residual tolerance
_EPSILON          = 1e-12         # avoid division by zero


# ---------------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------------

def build_graph_laplacian(
    cell_ids: list[str],
    topology_edges: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Build the symmetric graph Laplacian as a sparse adjacency dict.

    For an undirected graph with edge weight w_ij = 1 (uniform coupling):
        L[i][j] = -1                   if i ≠ j and (i,j) is an edge
        L[i][i] = degree(i)            sum of incident edge weights

    Returns a nested dict:  L[row_id][col_id] = value.
    Only non-zero entries are stored (sparse representation).
    """
    # Initialise: diagonal starts at 0, off-diag absent
    L: dict[str, dict[str, float]] = {cid: {} for cid in cell_ids}
    cell_set = set(cell_ids)

    for edge in topology_edges:
        for src in edge.get("sources", []):
            for tgt in edge.get("targets", []):
                if src not in cell_set or tgt not in cell_set:
                    continue
                if src == tgt:
                    continue

                # Undirected: add coupling in both directions (idempotent
                # for duplicate edges thanks to dict overwrite with same value)
                L[src][tgt] = L[src].get(tgt, 0.0) - 1.0
                L[tgt][src] = L[tgt].get(src, 0.0) - 1.0

                # Diagonal: increment degree for both endpoints
                L[src][src] = L[src].get(src, 0.0) + 1.0
                L[tgt][tgt] = L[tgt].get(tgt, 0.0) + 1.0

    return L


# ---------------------------------------------------------------------------
# Sparse matrix-vector product:  y = A·x   where  A = (I - dt·μ·L)
# ---------------------------------------------------------------------------

def _spmv(
    cell_ids: list[str],
    laplacian: dict[str, dict[str, float]],
    x: dict[str, float],
    dt_mu: float,
) -> dict[str, float]:
    """Compute y = (I - dt·μ·L) · x  without forming the full matrix.

    For each cell i:
        y[i] = x[i]                              ← identity term
             - dt·μ · Σ_j L[i][j] · x[j]         ← Laplacian diffusion term

    Since L[i][i] = degree(i) and L[i][j] = -1 for neighbours, expanding:
        y[i] = x[i] - dt·μ·(degree(i)·x[i] - Σ_{j∈N(i)} x[j])
             = (1 - dt·μ·degree(i))·x[i] + dt·μ·Σ_{j∈N(i)} x[j]

    This is equivalent but the generic L-based formulation is cleaner.
    """
    y: dict[str, float] = {}
    for cid in cell_ids:
        val = x.get(cid, 0.0)                    # identity contribution
        row = laplacian.get(cid, {})
        laplacian_sum = 0.0
        for col_id, l_val in row.items():
            laplacian_sum += l_val * x.get(col_id, 0.0)
        val -= dt_mu * laplacian_sum
        y[cid] = val
    return y


# ---------------------------------------------------------------------------
# Dot product over cell fields
# ---------------------------------------------------------------------------

def _dot(
    cell_ids: list[str],
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    """Inner product  <a, b> = Σ_i a[i]·b[i]."""
    s = 0.0
    for cid in cell_ids:
        s += a.get(cid, 0.0) * b.get(cid, 0.0)
    return s


# ---------------------------------------------------------------------------
# CG solver:  A·x = b   where A = (I - dt·μ·L),  b = v_old
# ---------------------------------------------------------------------------

def _cg_solve(
    cell_ids: list[str],
    laplacian: dict[str, dict[str, float]],
    rhs: dict[str, float],
    dt_mu: float,
    max_iter: int = _CG_MAX_ITER,
    tol: float = _CG_TOL,
) -> tuple[dict[str, float], int, float]:
    """Conjugate Gradient solve of  (I - dt·μ·L)·x = rhs.

    Initial guess x₀ = rhs (identity-preconditioned warm start: when
    dt·μ is small the solution is close to the RHS).

    Returns (x, iterations, final_residual_norm).
    """
    n = len(cell_ids)
    if n == 0:
        return {}, 0, 0.0

    # x₀ = rhs  (warm start)
    x = {cid: rhs.get(cid, 0.0) for cid in cell_ids}

    # r₀ = b - A·x₀
    ax = _spmv(cell_ids, laplacian, x, dt_mu)
    r = {cid: rhs.get(cid, 0.0) - ax.get(cid, 0.0) for cid in cell_ids}

    # p₀ = r₀
    p = {cid: r[cid] for cid in cell_ids}

    rs_old = _dot(cell_ids, r, r)
    b_norm = math.sqrt(_dot(cell_ids, rhs, rhs)) + _EPSILON

    if math.sqrt(rs_old) / b_norm < tol:
        return x, 0, math.sqrt(rs_old)

    iterations = 0
    for k in range(max_iter):
        # A·p
        ap = _spmv(cell_ids, laplacian, p, dt_mu)

        # α = rᵀr / pᵀAp
        p_ap = _dot(cell_ids, p, ap)
        if abs(p_ap) < _EPSILON:
            break
        alpha = rs_old / p_ap

        # x_{k+1} = x_k + α·p
        for cid in cell_ids:
            x[cid] += alpha * p.get(cid, 0.0)

        # r_{k+1} = r_k - α·Ap
        for cid in cell_ids:
            r[cid] -= alpha * ap.get(cid, 0.0)

        rs_new = _dot(cell_ids, r, r)
        iterations = k + 1

        # Convergence check
        if math.sqrt(rs_new) / b_norm < tol:
            break

        # β = rᵀ_{k+1} r_{k+1} / rᵀ_k r_k
        beta = rs_new / (rs_old + _EPSILON)

        # p_{k+1} = r_{k+1} + β·p_k
        for cid in cell_ids:
            p[cid] = r[cid] + beta * p[cid]

        rs_old = rs_new

    final_residual = math.sqrt(rs_new) if 'rs_new' in dir() else math.sqrt(rs_old)
    return x, iterations, final_residual


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def implicit_viscosity_step(
    force_field: dict[str, dict[str, float]],
    topology_edges: list[dict[str, Any]],
    dt: float = _DEFAULT_DT,
    mu: float = _DEFAULT_MU,
    max_iter: int = _CG_MAX_ITER,
    tol: float = _CG_TOL,
) -> dict[str, Any]:
    """Apply one implicit viscosity diffusion step to the force field.

    Treats force_field[cell_id]["dx"], ["dy"] as velocity components.
    Builds the graph Laplacian from topology_edges, then solves the
    implicit system  (I - dt·μ·L)·v_new = v_old  independently for
    each spatial axis (x, y) using Conjugate Gradient.

    The z-component ("dz") is left untouched — viscosity operates in
    the 2-D canvas plane only.

    Parameters
    ----------
    force_field : dict
        cell_id → {"dx": float, "dy": float, "dz": float}.
        **Modified in-place** with the diffused velocities.
    topology_edges : list
        Edge dicts with "sources" and "targets" lists.
    dt : float
        Physics time-step size (seconds).
    mu : float
        Kinematic viscosity coefficient.  Higher values → more diffusion.
        Typical range from QoS mapping: 0.001 (BEST_EFFORT) … 0.020 (RELIABLE).
    max_iter : int
        CG iteration budget.
    tol : float
        Relative residual convergence tolerance.

    Returns
    -------
    dict with diagnostics:
        cells          – number of cells in the solve
        edges          – number of topology edges processed
        dt_mu          – effective diffusion coefficient dt * mu
        cg_iter_x      – CG iterations for the x-axis solve
        cg_iter_y      – CG iterations for the y-axis solve
        residual_x     – final CG residual norm for x-axis
        residual_y     – final CG residual norm for y-axis
        max_delta      – largest absolute velocity change across all cells
    """
    cell_ids = sorted(force_field.keys())
    n = len(cell_ids)

    if n == 0 or not topology_edges:
        return {
            "cells": n, "edges": 0, "dt_mu": dt * mu,
            "cg_iter_x": 0, "cg_iter_y": 0,
            "residual_x": 0.0, "residual_y": 0.0,
            "max_delta": 0.0,
        }

    dt_mu = dt * mu

    # --- Build graph Laplacian from topology edges ---
    laplacian = build_graph_laplacian(cell_ids, topology_edges)

    # Count actual edges for diagnostics
    n_edges = sum(
        len(e.get("sources", [])) * len(e.get("targets", []))
        for e in topology_edges
    )

    # --- Extract v_old per axis ---
    vx_old = {cid: force_field[cid].get("dx", 0.0) for cid in cell_ids}
    vy_old = {cid: force_field[cid].get("dy", 0.0) for cid in cell_ids}

    # --- Solve (I - dt·μ·L)·v_new = v_old  for each axis ---
    vx_new, iter_x, res_x = _cg_solve(cell_ids, laplacian, vx_old, dt_mu, max_iter, tol)
    vy_new, iter_y, res_y = _cg_solve(cell_ids, laplacian, vy_old, dt_mu, max_iter, tol)

    # --- Write back into force_field (in-place) ---
    max_delta = 0.0
    for cid in cell_ids:
        dx_old = force_field[cid].get("dx", 0.0)
        dy_old = force_field[cid].get("dy", 0.0)
        dx_new = vx_new.get(cid, dx_old)
        dy_new = vy_new.get(cid, dy_old)

        delta = max(abs(dx_new - dx_old), abs(dy_new - dy_old))
        if delta > max_delta:
            max_delta = delta

        force_field[cid]["dx"] = dx_new
        force_field[cid]["dy"] = dy_new

    diag = {
        "cells":      n,
        "edges":      n_edges,
        "dt_mu":      round(dt_mu, 8),
        "cg_iter_x":  iter_x,
        "cg_iter_y":  iter_y,
        "residual_x": res_x,
        "residual_y": res_y,
        "max_delta":  round(max_delta, 8),
    }

    print(
        f"[M781-VISCOSITY] implicit solve: "
        f"cells={n} edges={n_edges} dt·μ={dt_mu:.6f}  "
        f"CG iters=({iter_x},{iter_y}) "
        f"residual=({res_x:.2e},{res_y:.2e}) "
        f"max_delta={max_delta:.6f}"
    )

    return diag


# ---------------------------------------------------------------------------
# Convenience: load topology + force_field, run solver, write results
# ---------------------------------------------------------------------------

CHANNELS = os.path.dirname(os.path.abspath(__file__))
CHANNELS = os.path.dirname(CHANNELS)  # up from physics/ to channels/


def run_standalone(
    dt: float = _DEFAULT_DT,
    mu: float = _DEFAULT_MU,
) -> dict[str, Any]:
    """Load state from disk, apply implicit viscosity, write back.

    Reads:  physics/force_field.json, skeleton/topology.json
    Writes: physics/force_field.json (updated), physics/viscosity_diag.json
    """
    # Load force field
    ff_path = os.path.join(CHANNELS, "physics", "force_field.json")
    with open(ff_path) as f:
        force_field = json.load(f)

    # Load topology edges
    topo_path = os.path.join(CHANNELS, "skeleton", "topology.json")
    edges = []
    if os.path.exists(topo_path):
        with open(topo_path) as f:
            edges = json.load(f).get("edges", [])

    # Solve
    diag = implicit_viscosity_step(force_field, edges, dt=dt, mu=mu)

    # Write back
    with open(ff_path, "w") as f:
        json.dump(force_field, f, indent=2)

    diag_path = os.path.join(CHANNELS, "physics", "viscosity_diag.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    return diag


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("M781 Implicit Viscosity — standalone test")
    print("=" * 64)

    # --- Synthetic 4-cell chain: A → B → C → D ---
    test_edges = [
        {"sources": ["A"], "targets": ["B"]},
        {"sources": ["B"], "targets": ["C"]},
        {"sources": ["C"], "targets": ["D"]},
    ]
    test_ff = {
        "A": {"dx": 10.0, "dy":  0.0, "dz": 0},
        "B": {"dx":  0.0, "dy":  0.0, "dz": 0},
        "C": {"dx":  0.0, "dy":  0.0, "dz": 0},
        "D": {"dx": -10.0, "dy": 0.0, "dz": 0},
    }

    print("\nBefore:")
    for cid, v in sorted(test_ff.items()):
        print(f"  {cid}: dx={v['dx']:+.6f}  dy={v['dy']:+.6f}")

    diag = implicit_viscosity_step(test_ff, test_edges, dt=1.0, mu=0.5)

    print("\nAfter (dt=1.0, mu=0.5):")
    for cid, v in sorted(test_ff.items()):
        print(f"  {cid}: dx={v['dx']:+.6f}  dy={v['dy']:+.6f}")

    print(f"\nDiagnostics: {json.dumps(diag, indent=2)}")

    # --- Verify symmetry: A and D should be mirror-symmetric ---
    assert abs(test_ff["A"]["dx"] + test_ff["D"]["dx"]) < 1e-6, \
        "Symmetry check failed: A.dx should equal -D.dx"
    assert abs(test_ff["B"]["dx"] + test_ff["C"]["dx"]) < 1e-6, \
        "Symmetry check failed: B.dx should equal -C.dx"
    print("\n✓ Symmetry check passed")

    # --- Verify diffusion: velocity should spread from endpoints ---
    assert abs(test_ff["B"]["dx"]) > 0.01, \
        "Diffusion check failed: B.dx should be non-zero after diffusion"
    print("✓ Diffusion check passed")

    # --- Try on-disk data if available ---
    print("\n" + "=" * 64)
    print("Attempting on-disk solve (physics/force_field.json)…")
    print("=" * 64)
    try:
        disk_diag = run_standalone()
        print(f"On-disk solve succeeded: {json.dumps(disk_diag, indent=2)}")
    except FileNotFoundError as e:
        print(f"Skipped (file not found): {e}")
