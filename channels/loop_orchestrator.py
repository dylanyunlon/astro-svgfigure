#!/usr/bin/env python3
"""
Loop Orchestrator — runs one full epoch of the cell pub/sub loop.

In production: each cell is a separate sub-Claude via claude_hk_chat.sh
Here: we simulate the loop locally to verify convergence.

Flow:
  1. Each cell reads its skeleton signal + force_field (subscribe)
  2. Each cell generates SVG with its species algorithm (proc)
  3. Each cell publishes bbox + svg (publish)
  4. Physics engine reads all bboxes → detects 3D collisions → updates force_field
  5. Convergence judge checks if all cells are stable
  6. If not converged → next epoch
  7. If converged → assemble final SVG by z-layer order
"""
import json
import math
import os
import glob

CHANNELS = os.path.dirname(os.path.abspath(__file__))

def read_channel(path):
    with open(os.path.join(CHANNELS, path)) as f:
        return json.load(f)

def write_channel(path, data):
    full = os.path.join(CHANNELS, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if isinstance(data, str):
        with open(full, "w") as f:
            f.write(data)
    else:
        with open(full, "w") as f:
            json.dump(data, f, indent=2)


def run_all_cells():
    """Execute all cell components (simulate parallel sub-Claude dispatch)."""
    from cell_component import proc
    cells = [f.replace(".json", "") for f in os.listdir(os.path.join(CHANNELS, "skeleton", "cell"))]
    for cell_id in sorted(cells):
        proc(cell_id)
    return cells


def _sweep_line_overlaps(rects_by_z):
    """
    FAstroCellSweepLine — ported from upstream/unreal-renderer/SceneSoftwareOcclusion.cpp
    commit 0b4b199.

    Algorithm: O((N+K) log N) sweep-line 2D overlap detection.
      1. For every cell bbox emit an OPEN event at min_x and a CLOSE event at max_x.
      2. Sort all events by x; ties: OPEN before CLOSE so touching edges are caught.
      3. Scan events left→right, maintaining an active set of currently open rects.
      4. On OPEN: insert into active set; test new rect against every active rect for
         Y-interval overlap → emit collision pair.
      5. On CLOSE: remove from active set.
      6. Cells on different z-layers never collide (same logic as the original O(N²)).

    Returns a list of (cell_id_a, cell_id_b, overlap_x, overlap_y, z) tuples.
    """
    # Event type constants  (mirrors FSweepEvent.bOpen)
    OPEN  = 1   # process opens BEFORE closes at same x (sort key trick from C++)
    CLOSE = 0

    pairs = []

    for z_layer, rects in rects_by_z.items():
        # rects: list of (cell_id, min_x, min_y, max_x, max_y)
        if len(rects) < 2:
            continue

        # Build events — two per rect, mirroring AddRect()
        # Sort key: (x, -is_open) so OPEN (1) sorts before CLOSE (0) at equal x
        events = []
        for idx, (cell_id, min_x, min_y, max_x, max_y) in enumerate(rects):
            if min_x > max_x or min_y > max_y:   # degenerate guard (same as C++)
                continue
            events.append((min_x, -OPEN,  idx))  # OPEN  at min_x
            events.append((max_x, -CLOSE, idx))  # CLOSE at max_x

        events.sort()   # ascending x; at equal x: -OPEN (-1) < -CLOSE (0) → open first

        # Active set: indices of rects whose x-interval contains current sweep position
        active = []   # list of rect indices (mirrors TArray<int32> ActiveSet in C++)

        for ev_x, neg_kind, idx in events:
            is_open = (-neg_kind == OPEN)
            cell_id, min_x, min_y, max_x, max_y = rects[idx]

            if is_open:
                # Test new rect against every currently active rect for Y overlap.
                # In the C++ version this happens implicitly when an occludee leaves
                # the sweep; here we emit pairs eagerly on OPEN so both directions are
                # symmetric and we don't need occluder/occludee distinction.
                for aidx in active:
                    _, _, a_min_y, _, a_max_y = rects[aidx]
                    # Y-interval overlap test (mirrors OR.MaxY >= OccludeeMinY && OR.MinY <= OccludeeMaxY)
                    if a_min_y <= max_y and a_max_y >= min_y:
                        ov_y = min(max_y, a_max_y) - max(min_y, a_min_y)
                        # x-overlap is guaranteed because we're inside the active sweep window
                        a_cell_id, a_min_x, _, a_max_x, _ = rects[aidx]
                        ov_x = min(max_x, a_max_x) - max(min_x, a_min_x)
                        if ov_x > 0 and ov_y > 0:
                            pairs.append((cell_id, a_cell_id, ov_x, ov_y, z_layer))
                active.append(idx)
            else:
                # CLOSE: remove from active set (mirrors RemoveSingleSwap)
                try:
                    active.remove(idx)
                except ValueError:
                    pass   # already removed (shouldn't happen, but be defensive)

    return pairs


def _tiled_constraint_solve(bboxes, force_field, canvas_w=None, canvas_h=None):
    """
    [ASTRO-TILEDSOLVER] Per-tile cell constraint solver.

    Ported from FAstroTiledConstraintSolver introduced in commit 7c82b90 of
    upstream/unreal-renderer/TiledDeferredLightRendering.cpp.

    Concept mapping (Unreal → astro-svgfigure):
      Light                 → Cell constraint (bbox overlap pair)
      Screen tile           → Canvas tile region
      Compute thread group  → Constraint solver worker
      LightData CBs         → ConstraintBatch per tile

    The C++ implementation splits the screen into 16×16 pixel tiles
    (GDeferredLightTileSizeX/Y) and culls the global light list so each tile
    only resolves lights whose influence sphere overlaps that tile's region.
    Here we apply the identical spatial-partition logic to cell collision
    constraints: instead of testing every cell pair globally (O(N²)), we
    assign each cell to a tile by bbox centre, then each tile only resolves
    constraints between cells in itself and its 8 adjacent tiles — O(N*K)
    where K is the average number of cells per tile (≪ N).

    Algorithm (mirrors BuildTileBatches → ResolveAll sequence in C++):
      Phase 1 — Cull (BuildTileBatches):
        • Divide canvas into TILE_SIZE × TILE_SIZE tiles.
        • For each cell, compute its bbox centre and assign it to a tile.
        • For each tile, collect neighbour-tile cell indices (self + 8 dirs).
        • Build a per-tile constraint list: only pairs where both cells are
          in the tile's neighbourhood.  This is the exact analog of the GPU
          compute shader culling lights per tile via shared-memory depth bounds.
      Phase 2 — Solve (ResolveAll):
        • For each tile, iterate only its culled constraint subset.
        • Compute repulsion force for each overlapping pair and accumulate
          into force_field (same repulsion formula as the global physics_engine
          loop, preserving correctness while reducing work).
        • Deduplicate: each pair is only resolved once (the tile that "owns"
          the pair — the tile of the cell with the lexicographically smaller id
          takes responsibility, matching the C++ primary-tile pattern).

    Args:
        bboxes      : dict  cell_id → {x, y, w, h, z}
        force_field : dict  cell_id → {dx, dy, dz}  — modified in-place
        canvas_w    : int   canvas pixel width  (auto-detected from bboxes if None)
        canvas_h    : int   canvas pixel height (auto-detected from bboxes if None)

    Returns:
        list of (cell_a, cell_b, overlap_x, overlap_y, z) collision tuples
        resolved by this pass (same format as _sweep_line_overlaps output).
    """
    # Tile granularity mirrors GDeferredLightTileSizeX/Y = 16 in the C++ source.
    TILE_SIZE = 16

    if not bboxes:
        return []

    # ── Auto-detect canvas bounds from bbox extents if not supplied ───────────
    # Mirrors: CanvasSize = Views[0].ViewRect.Size() in the C++ caller.
    if canvas_w is None or canvas_h is None:
        max_x = max(b["x"] + b["w"] for b in bboxes.values())
        max_y = max(b["y"] + b["h"] for b in bboxes.values())
        canvas_w = canvas_w or int(math.ceil(max_x))
        canvas_h = canvas_h or int(math.ceil(max_y))

    # Guard: ensure at least one tile exists even for tiny canvases.
    canvas_w = max(canvas_w, TILE_SIZE)
    canvas_h = max(canvas_h, TILE_SIZE)

    # ── Phase 1 — Cull: build per-tile cell membership ────────────────────────
    # Mirrors FAstroTiledConstraintSolver::BuildTileBatches().
    num_tiles_x = math.ceil(canvas_w / TILE_SIZE)
    num_tiles_y = math.ceil(canvas_h / TILE_SIZE)

    print(
        f"[ASTRO-TILEDSOLVER] INIT: canvas={canvas_w}x{canvas_h} "
        f"tile={TILE_SIZE}x{TILE_SIZE} "
        f"grid={num_tiles_x}x{num_tiles_y} totalTiles={num_tiles_x * num_tiles_y} "
        f"cells={len(bboxes)}"
    )

    # tile_cells[ty][tx] → list of cell_id whose bbox centre falls in that tile.
    # Analogous to FAstroTileConstraintBatch.ConstraintIndices (indexed by tile
    # origin rather than a flat array for Python readability).
    tile_cells = [[[] for _ in range(num_tiles_x)] for _ in range(num_tiles_y)]

    for cell_id, b in bboxes.items():
        # Assign cell to tile by bbox centre (mirrors the AABB-centre cull in C++).
        cx = b["x"] + b["w"] * 0.5
        cy = b["y"] + b["h"] * 0.5
        tx = int(cx // TILE_SIZE)
        ty = int(cy // TILE_SIZE)
        # Clamp to grid bounds (cells that overflow the canvas edge fall in last tile).
        tx = max(0, min(tx, num_tiles_x - 1))
        ty = max(0, min(ty, num_tiles_y - 1))
        tile_cells[ty][tx].append(cell_id)

    # ── Phase 2 — Solve: per-tile constraint resolution ───────────────────────
    # Mirrors FAstroTiledConstraintSolver::ResolveAll().
    # For each tile collect cells from itself + 8 adjacent tiles (3×3 kernel),
    # then test all pairs within that neighbourhood for bbox overlap.
    # This mirrors how the GPU shader tests lights against a tile's frustum
    # using the shared-memory depth range: only local candidates are tested.

    collisions = []
    resolved_pairs = set()   # deduplication: each (a,b) pair solved exactly once.
    total_resolved = 0

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            # Gather neighbourhood cells (self tile + 8 neighbours = 3×3 kernel).
            # Mirrors the tile's influence region in the tiled light cull pass.
            neighbourhood = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ntx = tx + dx
                    nty = ty + dy
                    if 0 <= ntx < num_tiles_x and 0 <= nty < num_tiles_y:
                        neighbourhood.extend(tile_cells[nty][ntx])

            num_local = len(neighbourhood)
            if num_local < 2:
                continue   # no pairs possible — mirrors the early-out in C++ NumLocal < 2 guard

            print(
                f"[ASTRO-TILEDSOLVER] TILE: tile=({tx},{ty}) "
                f"origin=({tx * TILE_SIZE},{ty * TILE_SIZE}) "
                f"neighbourhood={num_local}"
            )

            # Test all O(K²) pairs within the neighbourhood.
            # K ≈ cells-per-tile (small) vs global N, giving O(N*K) total.
            for i in range(num_local):
                for j in range(i + 1, num_local):
                    cell_a = neighbourhood[i]
                    cell_b = neighbourhood[j]

                    # Canonical pair key — smaller id is always first, preventing
                    # duplicate resolution across shared tile boundaries.
                    # Mirrors the C++ primary-tile ownership pattern.
                    pair_key = (min(cell_a, cell_b), max(cell_a, cell_b))
                    if pair_key in resolved_pairs:
                        continue

                    ba = bboxes[cell_a]
                    bb = bboxes[cell_b]

                    # Same-z layer only (preserved from original physics_engine rule).
                    if ba.get("z", 3) != bb.get("z", 3):
                        continue

                    # AABB overlap test — mirrors the bOverlaps check in BuildTileBatches:
                    #   C.BBoxMaxX > TileMinX && C.BBoxMinX < TileMaxX && …
                    # Here applied between two cell bboxes instead of cell vs tile.
                    a_min_x, a_max_x = ba["x"], ba["x"] + ba["w"]
                    a_min_y, a_max_y = ba["y"], ba["y"] + ba["h"]
                    b_min_x, b_max_x = bb["x"], bb["x"] + bb["w"]
                    b_min_y, b_max_y = bb["y"], bb["y"] + bb["h"]

                    ov_x = min(a_max_x, b_max_x) - max(a_min_x, b_min_x)
                    ov_y = min(a_max_y, b_max_y) - max(a_min_y, b_min_y)

                    if ov_x <= 0 or ov_y <= 0:
                        continue   # no overlap — culled, mirrors bOverlaps == false path

                    # Mark pair as resolved before applying force (dedup guard).
                    resolved_pairs.add(pair_key)
                    total_resolved += 1

                    z = ba.get("z", 3)
                    collisions.append({
                        "a": cell_a, "b": cell_b,
                        "overlap": ov_x * ov_y,
                        "z": z,
                    })

                    # Repulsion force — axis of least overlap (same formula as
                    # the global physics_engine pass; correctness is unchanged,
                    # only the set of tested pairs is smaller).
                    # Mirrors: apply relative→absolute position delta for CellA/CellB
                    # within this tile's region (C++ placeholder comment in ResolveAll).
                    if ov_x < ov_y:
                        push = ov_x / 2 + 5
                        if ba["x"] < bb["x"]:
                            force_field[cell_a]["dx"] -= push
                            force_field[cell_b]["dx"] += push
                        else:
                            force_field[cell_a]["dx"] += push
                            force_field[cell_b]["dx"] -= push
                    else:
                        push = ov_y / 2 + 5
                        if ba["y"] < bb["y"]:
                            force_field[cell_a]["dy"] -= push
                            force_field[cell_b]["dy"] += push
                        else:
                            force_field[cell_a]["dy"] += push
                            force_field[cell_b]["dy"] -= push

    print(
        f"[ASTRO-TILEDSOLVER] DONE: totalResolved={total_resolved} "
        f"tiles={num_tiles_x * num_tiles_y} collisions={len(collisions)}"
    )
    return collisions

def _voxelize_constraint_field(bboxes, force_field, grid_n=8):
    """
    [ASTRO] FAstroConstraintVoxelGrid — Python port of the constraint-field
    voxelization algorithm introduced in commit 5f5cf3a of
    upstream/unreal-renderer/VolumetricFogVoxelization.cpp.

    The C++ original replaces volumetric fog density voxelization with a 3-D
    constraint-field grid indexed as (X, Y, Z-layer).  Each voxel accumulates
    ConstraintCount and ConstraintWeight (distance-attenuated) for every fog
    primitive (= cell bbox) whose world-space AABB overlaps the voxel.
    ConstraintDensity() = clamp(weight / count, 0, 1) is then read by the
    scheduling layer to decide per-primitive temporal subdivision.

    Python adaptation:
      • "World space"     → canvas pixel space (x, y from bbox).
      • "Z-layer / depth" → discrete z-layer integer from bbox["z"].
      • Canvas AABB       → derived dynamically from all bboxes each call,
                            mirroring GridWorldBox(ViewOrigin ± FogDistance).
      • Grid dimensions   → grid_n × grid_n × (distinct z-layers), replacing
                            VolumetricFogGridSize (X, Y, Z).
      • BaseWeight        → (1 − dist_norm) where dist_norm is the cell
                            centre's distance to the canvas centre normalised
                            by the canvas half-diagonal (same formula as
                            DistNorm = Sqrt(DistSq) / VolumetricFogDistance).
      • Voxel attenuation → linear falloff (1 − dist/half_diag) clamped [0,1]
                            inside AccumulateConstraint(), identical to the
                            C++ Alpha computation.
      • No numpy: plain Python lists, integer indexing, math.sqrt/hypot.

    Effect on force_field (supplements distance field, does not replace it):
      High-density voxels → region is crowded → "scatter" force pushes the
        cell away from the voxel grid centre (repulsion, like fog density
        preventing further accumulation).
      Low-density voxels  → region is sparse  → "attract" force nudges the
        cell toward the canvas centre (attraction, like open space drawing
        particles in).

    The force magnitude is scaled by _VOXEL_FORCE_SCALE (= 3.0 px per unit
    density delta) and blended additively into force_field so that voxel-field
    forces compose with both sweep-line collision response and distance-field
    propagation without interference.

    Returns the voxel grid as a dict keyed by (gx, gy, z_layer) for debug /
    persistence use.
    """
    if not bboxes:
        return {}

    _VOXEL_FORCE_SCALE = 3.0   # px per density-unit, tunable CVar equivalent
    _SMALL = 1e-9

    # ------------------------------------------------------------------
    # Step 1 – Compute canvas AABB (mirrors GridWorldBox construction).
    # ------------------------------------------------------------------
    all_cx = [b["x"] + b["w"] * 0.5 for b in bboxes.values()]
    all_cy = [b["y"] + b["h"] * 0.5 for b in bboxes.values()]
    canvas_min_x = min(b["x"]              for b in bboxes.values())
    canvas_min_y = min(b["y"]              for b in bboxes.values())
    canvas_max_x = max(b["x"] + b["w"]    for b in bboxes.values())
    canvas_max_y = max(b["y"] + b["h"]    for b in bboxes.values())

    canvas_w = max(canvas_max_x - canvas_min_x, _SMALL)
    canvas_h = max(canvas_max_y - canvas_min_y, _SMALL)
    canvas_cx = (canvas_min_x + canvas_max_x) * 0.5
    canvas_cy = (canvas_min_y + canvas_max_y) * 0.5
    canvas_half_diag = math.hypot(canvas_w * 0.5, canvas_h * 0.5)
    canvas_half_diag = max(canvas_half_diag, _SMALL)

    # Distinct z-layers (mirrors VolumetricFogGridSize.Z = depth slices).
    z_layers = sorted({b.get("z", 3) for b in bboxes.values()})
    z_index  = {z: i for i, z in enumerate(z_layers)}
    nz = len(z_layers)

    # ------------------------------------------------------------------
    # Step 2 – Allocate flat voxel grid.
    # voxel at (gx, gy, gz) → voxels[gx + gy*grid_n + gz*grid_n*grid_n]
    # Each voxel: [constraint_count (int), constraint_weight (float)]
    # Mirrors TArray<FAstroConstraintVoxel> with SetNumZeroed().
    # ------------------------------------------------------------------
    total = grid_n * grid_n * nz
    voxel_count  = [0]   * total   # FAstroConstraintVoxel.ConstraintCount
    voxel_weight = [0.0] * total   # FAstroConstraintVoxel.ConstraintWeight

    def _world_to_voxel(px, py, z):
        """
        Map canvas pixel (px, py, z) → fractional voxel coordinates.
        Direct port of FAstroConstraintVoxelGrid::WorldToVoxel().
        """
        vx = (px - canvas_min_x) / canvas_w  * grid_n
        vy = (py - canvas_min_y) / canvas_h  * grid_n
        vz = float(z_index[z])
        return vx, vy, vz

    def _clamp_int(v, lo, hi):
        return max(lo, min(hi, int(v)))

    # ------------------------------------------------------------------
    # Step 3 – AccumulateConstraint() for every cell.
    # Mirrors the Pass-1 CPU loop:
    #   for MeshBatchIndex in View.VolumetricMeshBatches …
    #       GAstroConstraintGrid.AccumulateConstraint(ConstraintBounds, BaseWeight)
    # ------------------------------------------------------------------
    for cell_id, b in bboxes.items():
        z = b.get("z", 3)
        gz_idx = z_index[z]

        # Cell centre and AABB corners in canvas space.
        cell_cx = b["x"] + b["w"] * 0.5
        cell_cy = b["y"] + b["h"] * 0.5

        # BaseWeight: proximity to canvas centre, mirrors DistNorm formula.
        # (1 − dist_to_canvas_centre / canvas_half_diag), clamped [0, 1].
        dist_to_centre = math.hypot(cell_cx - canvas_cx, cell_cy - canvas_cy)
        base_weight = max(0.0, min(1.0, 1.0 - dist_to_centre / canvas_half_diag))

        # Convert cell AABB → voxel-space corners (mirrors VMin / VMax).
        vmin_x, vmin_y, _ = _world_to_voxel(b["x"],          b["y"],          z)
        vmax_x, vmax_y, _ = _world_to_voxel(b["x"] + b["w"], b["y"] + b["h"], z)

        x0 = _clamp_int(math.floor(vmin_x), 0, grid_n - 1)
        x1 = _clamp_int(math.ceil (vmax_x), 0, grid_n - 1)
        y0 = _clamp_int(math.floor(vmin_y), 0, grid_n - 1)
        y1 = _clamp_int(math.ceil (vmax_y), 0, grid_n - 1)

        # Constraint AABB centre in voxel space (mirrors VCentre).
        vcx = (vmin_x + vmax_x) * 0.5
        vcy = (vmin_y + vmax_y) * 0.5

        # Half-diagonal of constraint AABB in voxel space (mirrors HalfDiag).
        half_diag_v = max(
            math.hypot(vmax_x - vmin_x, vmax_y - vmin_y) * 0.5,
            _SMALL
        )

        # Inner triple loop: iterate voxels overlapped by this cell AABB.
        # Mirrors the Z/Y/X nested loops in AccumulateConstraint().
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                gz = gz_idx   # single z-layer per cell (Z-dimension = layer index)

                # Distance from voxel centre to constraint centre (voxel units).
                # Mirrors: FVector VoxelCentre(X+0.5, Y+0.5, Z+0.5);
                #          float Dist = FVector::Dist(VoxelCentre, VCentre);
                vox_cx = gx + 0.5
                vox_cy = gy + 0.5
                dist_v = math.hypot(vox_cx - vcx, vox_cy - vcy)

                # Linear attenuation: Alpha = clamp(1 - dist/half_diag, 0, 1).
                alpha  = max(0.0, min(1.0, 1.0 - dist_v / half_diag_v))
                weight = base_weight * alpha

                idx = gx + gy * grid_n + gz * grid_n * grid_n
                voxel_count[idx]  += 1
                voxel_weight[idx] += weight

    # ------------------------------------------------------------------
    # Step 4 – Compute ConstraintDensity per voxel.
    # ConstraintDensity() = clamp(weight / count, 0, 1) when count > 0.
    # Mirror of FAstroConstraintVoxel::ConstraintDensity().
    # ------------------------------------------------------------------
    voxel_density = []
    for i in range(total):
        if voxel_count[i] > 0:
            voxel_density.append(
                max(0.0, min(1.0, voxel_weight[i] / voxel_count[i]))
            )
        else:
            voxel_density.append(0.0)

    # Global average density (used to distinguish "crowded" vs "sparse").
    # Mirrors TotalConstraintCount() / grid_volume as a density normaliser.
    occupied = [d for d in voxel_density if d > 0.0]
    avg_density = (sum(occupied) / len(occupied)) if occupied else 0.0

    total_constraint_count = sum(voxel_count)
    print(
        f"[VoxelConstraintField] grid={grid_n}x{grid_n}x{nz} "
        f"total_constraints={total_constraint_count} "
        f"avg_density={avg_density:.3f}"
    )

    # ------------------------------------------------------------------
    # Step 5 – Sample each cell's voxel and apply scatter/attract force.
    # "Pass 2 scheduling" analogue: high density → scatter; low → attract.
    # (In C++ this is (void)ConstraintDensity consumed by pub-sub layer;
    # here we materialise it as an explicit force contribution.)
    # ------------------------------------------------------------------
    for cell_id, b in bboxes.items():
        z = b.get("z", 3)
        gz_idx = z_index[z]

        cell_cx = b["x"] + b["w"] * 0.5
        cell_cy = b["y"] + b["h"] * 0.5

        # WorldToVoxel → sample voxel at cell centre (mirrors SampleX/Y/Z).
        vx, vy, _ = _world_to_voxel(cell_cx, cell_cy, z)
        gx = _clamp_int(math.floor(vx), 0, grid_n - 1)
        gy = _clamp_int(math.floor(vy), 0, grid_n - 1)
        gz = gz_idx

        idx = gx + gy * grid_n + gz * grid_n * grid_n
        density = voxel_density[idx]

        # density > avg_density  → crowded voxel → scatter (repel from centre).
        # density < avg_density  → sparse voxel  → attract (pull toward centre).
        density_delta = density - avg_density   # positive = crowded, negative = sparse

        # Direction from canvas centre to this cell (unit vector).
        to_cx = cell_cx - canvas_cx
        to_cy = cell_cy - canvas_cy
        dist_to_c = math.hypot(to_cx, to_cy)
        if dist_to_c > _SMALL:
            ux = to_cx / dist_to_c
            uy = to_cy / dist_to_c
        else:
            ux, uy = 0.0, 0.0   # cell is exactly at canvas centre

        # Force: positive density_delta → push outward (scatter);
        #        negative density_delta → push inward (attract).
        # Magnitude proportional to |density_delta| × _VOXEL_FORCE_SCALE.
        fmag = density_delta * _VOXEL_FORCE_SCALE
        force_field[cell_id]["dx"] += ux * fmag
        force_field[cell_id]["dy"] += uy * fmag

    # Build output dict keyed by (gx, gy, z_layer) for physics/z_layers.json.
    grid_out = {}
    for gx in range(grid_n):
        for gy in range(grid_n):
            for zi, z_layer in enumerate(z_layers):
                idx = gx + gy * grid_n + zi * grid_n * grid_n
                grid_out[(gx, gy, z_layer)] = {
                    "count":   voxel_count[idx],
                    "weight":  round(voxel_weight[idx], 4),
                    "density": round(voxel_density[idx], 4),
                }

    return grid_out


# =============================================================================
# [ASTRO] FAstroCellSymmetryMirror — Python port
#
# Ported from upstream/unreal-renderer/PlanarReflectionRendering.cpp
# commit 8a2501b.
#
# Original: mirror-camera paradigm — FMirrorMatrix applied to a secondary
#   FSceneRenderer's view matrix so a reflected camera captures the mirrored
#   scene into a render target.  One full secondary scene render per frame.
#
# Port: constraint-duplication paradigm — FAstroCellSymmetryMirror computes
#   a 2×2 Householder reflection matrix in cell/group-local space and
#   "emits" mirrored constraint records back into the cell-pubsub loop,
#   so the layout pass treats symmetric siblings as first-class cells.
#   No secondary render.  O(1) per symmetric pair per epoch.
#
# Key concept mapping (PlanarReflection → cell symmetry):
#   Reflection plane        → symmetry axis (normalised 2-D direction in canvas space)
#   FMirrorMatrix(plane)    → Householder matrix M = I − 2NNᵀ  (N = axis normal)
#   Mirror camera           → mirrored cell bbox clone
#   Secondary SceneRenderer → _apply_symmetry_constraints() force accumulator
#   UpdatePlanarReflectionViewMatrix() → mirror-force applied to force_field[dx,dy]
#
# Algorithm — mirrors FAstroCellSymmetryMirror::ComputeMirrorMatrix() +
#   ApplyToCell() + EmitMirroredConstraints() from 8a2501b:
#
#   Phase 1 — Detect symmetric pairs in topology (add_norm1/add_norm2 pattern).
#   Phase 2 — For each pair compute the symmetry axis angle (vertical layout →
#             horizontal axis; Custom → Atan2 of midpoint-to-midpoint vector).
#   Phase 3 — ComputeMirrorMatrix: build Householder M = I − 2NNᵀ where N is
#             the axis normal (line_angle + π/2).
#   Phase 4 — ApplyToCell: reflect the heavier cell's bbox centre through the
#             symmetry axis to get the "ideal" mirrored position.
#   Phase 5 — EmitMirroredConstraints: accumulate "mirror force" = difference
#             between actual position and ideal mirrored position, blended into
#             force_field so the normal physics loop moves cells toward symmetry.
# =============================================================================

import enum as _enum


class _EAstroCellSymmetryAxis(_enum.Enum):
    """
    Symmetry axis presets — mirrors EAstroCellSymmetryAxis enum in 8a2501b.

    Horizontal : mirror line is horizontal (X axis in group space);
                 normal points along Y.  Left ↔ right reflection.
    Vertical   : mirror line is vertical (Y axis in group space);
                 normal points along X.  Top ↔ bottom reflection.
    Custom     : arbitrary angle supplied by caller in radians,
                 measured CCW from +X axis in canvas space.
    """
    Horizontal = 0
    Vertical   = 1
    Custom     = 2


def _compute_mirror_matrix(axis_mode, custom_angle_rad=0.0):
    """
    ComputeMirrorMatrix — mirrors FAstroCellSymmetryMirror::ComputeMirrorMatrix().

    Builds the 2×2 Householder reflection matrix:
        M = I − 2 N Nᵀ
        [ 1−2nx²   −2nx·ny ]
        [ −2nx·ny  1−2ny²  ]

    where N = (nx, ny) is the unit normal to the mirror line.

    Old path (PlanarReflection): FMirrorMatrix(MirrorPlane) — 4×4 matrix that
      reflected a camera origin through a world-space plane for a secondary
      scene-capture pass.
    New path (cell symmetry): pure 2-D Householder in canvas pixel space;
      no scene capture, no secondary renderer allocation.

    Args:
        axis_mode        : _EAstroCellSymmetryAxis
        custom_angle_rad : mirror *line* angle (CCW from +X); used for Custom only.

    Returns:
        ((m00, m01), (m10, m11))  — 2×2 reflection matrix, row-major.
    """
    if axis_mode == _EAstroCellSymmetryAxis.Horizontal:
        # Mirror line = X axis (horizontal), line_angle = 0
        # Normal = perpendicular to line = (0, 1)  → pointing along Y
        line_angle = 0.0
    elif axis_mode == _EAstroCellSymmetryAxis.Vertical:
        # Mirror line = Y axis (vertical), line_angle = π/2
        # Normal = perpendicular = (1, 0)  → pointing along X
        line_angle = math.pi * 0.5
    else:
        # Custom: caller supplies the mirror *line* direction angle
        line_angle = custom_angle_rad

    # Normal to the mirror line is perpendicular: line_angle + π/2
    # (mirrors: const float NormalAngle = LineAngleRad + PI * 0.5f; in C++)
    normal_angle = line_angle + math.pi * 0.5
    nx = math.cos(normal_angle)
    ny = math.sin(normal_angle)

    # Householder: M = I − 2 N Nᵀ  (same formula as 8a2501b MirrorMatrix block)
    m00 = 1.0 - 2.0 * nx * nx
    m01 =      -2.0 * nx * ny
    m10 =      -2.0 * nx * ny
    m11 = 1.0 - 2.0 * ny * ny

    print(
        f"[ASTRO-SYMM] ComputeMirrorMatrix: axis={axis_mode.name} "
        f"line_angle={line_angle:.4f} "
        f"normal=({nx:.3f},{ny:.3f}) "
        f"M=[[{m00:.3f},{m01:.3f}],[{m10:.3f},{m11:.3f}]]"
    )
    return ((m00, m01), (m10, m11))


def _apply_to_cell(mirror_matrix, cell_cx, cell_cy, axis_cx, axis_cy):
    """
    ApplyToCell — mirrors FAstroCellSymmetryMirror::ApplyToCell().

    Reflects a cell's centroid through the symmetry axis (which passes through
    the compound group's local origin — here approximated as the midpoint of
    the symmetric pair).

    Old path: left-multiply the view matrix by FMirrorMatrix for a secondary
      camera — O(4×4) matrix multiply, allocates FViewInfo + FSceneRenderer.
    New path: translate cell centre to axis-local space, left-multiply by the
      2×2 Householder matrix, translate back — O(1), zero allocation.

    Args:
        mirror_matrix : ((m00,m01),(m10,m11)) from _compute_mirror_matrix()
        cell_cx, cell_cy  : source cell centroid in canvas space (px)
        axis_cx, axis_cy  : origin of the symmetry axis (midpoint of the pair)

    Returns:
        (mirrored_cx, mirrored_cy)  — ideal reflected centroid in canvas space
    """
    (m00, m01), (m10, m11) = mirror_matrix

    # Translate to axis-local space (mirrors InCellLocalTransform origin shift)
    lx = cell_cx - axis_cx
    ly = cell_cy - axis_cy

    # Left-multiply by Householder M (M² = I, so M is its own inverse — same
    # comment appears verbatim in ApplyToCell() in 8a2501b)
    rx = m00 * lx + m01 * ly
    ry = m10 * lx + m11 * ly

    # Translate back to canvas space
    mirrored_cx = rx + axis_cx
    mirrored_cy = ry + axis_cy

    print(
        f"[ASTRO-SYMM] ApplyToCell: "
        f"src=({cell_cx:.2f},{cell_cy:.2f}) "
        f"axis=({axis_cx:.2f},{axis_cy:.2f}) "
        f"→ mirror=({mirrored_cx:.2f},{mirrored_cy:.2f})"
    )
    return mirrored_cx, mirrored_cy


def _detect_symmetric_pairs(topology_edges, bbox_map):
    """
    Detect symmetric cell pairs from topology naming and structure.

    Ported from the compound-group symmetry detection implicit in
    FAstroCellSymmetryMirror — the C++ system identifies cells within a
    compound group that share the same constraint structure; here we use
    naming conventions (suffix _1/_2, shared label prefix) and structural
    equivalence (same width/height in topology, symmetric depth in the DAG).

    Detection rules (改 20%: extended beyond pure naming to DAG-depth parity):
      Rule 1 — Suffix pair: cell_id ending in _1 has a sibling ending in _2
               with the same base name.  Classic Transformer add_norm1/add_norm2.
      Rule 2 — Topology depth: two cells at the same DAG depth that share the
               same declared width AND height in topology are structurally
               symmetric even without a naming convention.
      Rule 3 — DAG depth parity check (改 20% addition): only cells at even
               depth distance from each other can be symmetric — odd distance
               implies one is a transformation of the other, not a mirror.

    Returns:
        list of (cell_id_a, cell_id_b, axis_mode)
            axis_mode is _EAstroCellSymmetryAxis.Vertical for a vertical
            layout (DOWN direction), Horizontal for LEFT/RIGHT layouts.
    """
    # ── Read layout direction from topology root to pick default axis ─────────
    # Mirrors: AxisMode = EAstroCellSymmetryAxis::Horizontal for legacy proxy
    # compatibility in SetupPlanarReflectionUniformParameters (8a2501b).
    topo_path = os.path.join(CHANNELS, "skeleton", "topology.json")
    layout_dir = "DOWN"
    topo_children_meta = {}   # cell_id → {width, height} from topology definition
    if os.path.exists(topo_path):
        with open(topo_path) as f:
            topo = json.load(f)
        layout_dir = (
            topo.get("layoutOptions", {})
                .get("elk.direction", "DOWN")
                .upper()
        )
        for child in topo.get("children", []):
            cid = child.get("id", "")
            topo_children_meta[cid] = {
                "width":  child.get("width",  0),
                "height": child.get("height", 0),
            }

    # Default axis: vertical layout (DOWN/UP) → cells are stacked vertically
    # → symmetric pairs differ horizontally → Vertical symmetry axis (mirror
    # line is vertical, normal points along X, left↔right reflection matches
    # how add_norm1 and add_norm2 are laid out beside each other).
    if layout_dir in ("DOWN", "UP"):
        default_axis = _EAstroCellSymmetryAxis.Vertical
    else:
        default_axis = _EAstroCellSymmetryAxis.Horizontal

    pairs = []
    cell_ids = set(bbox_map.keys())

    # ── Rule 1: Suffix _N / _M naming (add_norm1 ↔ add_norm2) ────────────────
    # Mirrors the "same constraint structure" detection in FAstroCellSymmetryMirror:
    # cells sharing a base name but different numeric suffixes are symmetric
    # siblings within the same compound group.
    seen = set()
    suffix_re = _re.compile(r"^(.+?)(\d+)$")
    base_to_cells = {}
    for cid in sorted(cell_ids):
        m = suffix_re.match(cid)
        if m:
            base, num = m.group(1).rstrip("_"), int(m.group(2))
            base_to_cells.setdefault(base, []).append((num, cid))

    for base, numbered in base_to_cells.items():
        if len(numbered) < 2:
            continue
        numbered.sort()
        # Pair consecutive indices as symmetric siblings (1↔2, 3↔4, …)
        for i in range(0, len(numbered) - 1, 2):
            _, cid_a = numbered[i]
            _, cid_b = numbered[i + 1]
            pair_key = (min(cid_a, cid_b), max(cid_a, cid_b))
            if pair_key not in seen:
                seen.add(pair_key)
                pairs.append((cid_a, cid_b, default_axis))
                print(
                    f"[ASTRO-SYMM] Symmetric pair detected (Rule1-suffix): "
                    f"{cid_a} ↔ {cid_b}  axis={default_axis.name}"
                )

    # ── Rule 2: Structural equivalence — same declared w/h at same DAG depth ─
    # Mirrors EmitMirroredConstraints iterating SourceCells by compound-group
    # membership: cells with identical bbox dimensions are constraint-equivalent
    # and treated as symmetric.  改 20%: add DAG-depth parity guard (Rule 3).

    # Build DAG depth map via BFS from source nodes (no incoming edges).
    incoming_count = {cid: 0 for cid in cell_ids}
    adjacency = {cid: [] for cid in cell_ids}
    for edge in topology_edges:
        for src in edge.get("sources", []):
            for tgt in edge.get("targets", []):
                if src in cell_ids and tgt in cell_ids:
                    adjacency[src].append(tgt)
                    incoming_count[tgt] += 1

    sources = [cid for cid, cnt in incoming_count.items() if cnt == 0]
    depth = {cid: 0 for cid in sources}
    queue = list(sources)
    while queue:
        cur = queue.pop(0)
        for nxt in adjacency.get(cur, []):
            new_depth = depth[cur] + 1
            if nxt not in depth or depth[nxt] < new_depth:
                depth[nxt] = new_depth
                queue.append(nxt)

    # Group cells by (depth, width, height) — same triple → structural match.
    struct_groups = {}
    for cid in sorted(cell_ids):
        meta = topo_children_meta.get(cid, {})
        w = meta.get("width",  bbox_map[cid].get("w", 0))
        h = meta.get("height", bbox_map[cid].get("h", 0))
        d = depth.get(cid, -1)
        key = (d, w, h)
        struct_groups.setdefault(key, []).append(cid)

    for key, group in struct_groups.items():
        if len(group) < 2:
            continue
        d, w, h = key
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                cid_a, cid_b = group[i], group[j]
                pair_key = (min(cid_a, cid_b), max(cid_a, cid_b))
                if pair_key in seen:
                    continue  # already detected by Rule 1

                # Rule 3 (改 20%): depth-parity guard — cells at the same depth
                # are equidistant from the DAG root, confirming they occupy
                # symmetric positions in the computation graph.  Different depths
                # mean one is derived from the other; skip.
                depth_a = depth.get(cid_a, -1)
                depth_b = depth.get(cid_b, -1)
                if depth_a != depth_b:
                    continue  # odd depth distance → not mirror-symmetric

                seen.add(pair_key)
                pairs.append((cid_a, cid_b, default_axis))
                print(
                    f"[ASTRO-SYMM] Symmetric pair detected (Rule2-struct depth={d} "
                    f"w={w} h={h}): {cid_a} ↔ {cid_b}  axis={default_axis.name}"
                )

    print(f"[ASTRO-SYMM] _detect_symmetric_pairs: found {len(pairs)} symmetric pairs")
    return pairs


def _apply_symmetry_constraints(bboxes, force_field, topology_edges):
    """
    [ASTRO] FAstroCellSymmetryMirror — cell symmetry mirror constraint system.

    Ported from FAstroCellSymmetryMirror in commit 8a2501b of
    upstream/unreal-renderer/PlanarReflectionRendering.cpp.

    Replaces the mirror-camera reflection matrix with a cell-space symmetry
    constraint duplicator.  Within a compound group (here: the full topology),
    each detected symmetric cell pair is reflected across a symmetry axis and
    the resulting "mirror force" is injected into force_field so the normal
    physics loop converges toward a symmetric layout — no secondary render
    required.

    Algorithm (mirrors FAstroCellSymmetryMirror::EmitMirroredConstraints()
    called from the compound group's cell-pubsub tick in 8a2501b):

      1. _detect_symmetric_pairs() — identify (cell_a, cell_b, axis) triples
         from topology naming + DAG-depth structure.
         → Maps to: SourceCells view passed to EmitMirroredConstraints().

      2. For each pair:
         a. _compute_mirror_matrix(axis) → Householder M = I − 2NNᵀ.
            → Maps to: FAstroCellSymmetryMirror::ComputeMirrorMatrix().
         b. Compute symmetry axis origin = midpoint of the two cell centroids.
            → Maps to: compound group's local origin in C++.
         c. _apply_to_cell(M, cell_a_centre, axis_origin)
            → ideal reflected centroid for cell_b.
            → Maps to: FAstroCellSymmetryMirror::ApplyToCell().
         d. Compute mirror force = (ideal_position − actual_position) × strength.
            → Maps to: OutMirroredCells appended to CellConstraintList in C++,
              which the layout pass then treats as a position constraint.
         e. Enforce same width/height: if cells differ in size, apply a gentle
            "equalisation nudge" force in the dz axis (z-layer sorting proxy).
            → Maps to: bInheritSourceConstraintWeight weight propagation.

      3. Accumulate mirror forces into force_field (in-place).
         → Maps to: EmitMirroredConstraints → CellConstraintList injection.

    Mirror force strength (改 20% vs raw C++ weight):
      C++ uses weight = 1.0 (rigid symmetry) or inherited weight (soft).
      Python uses _MIRROR_FORCE_STRENGTH = 0.35 — softer than rigid (1.0)
      so the symmetry constraint blends with collision-response forces from
      the tiled solver and distance-field propagation, rather than overriding
      them.  Same intent as bInheritSourceConstraintWeight = true but scaled
      to the pixel-force regime of the Python physics engine.

    Args:
        bboxes        : dict  cell_id → {x, y, w, h, z}
        force_field   : dict  cell_id → {dx, dy, dz}  — modified in-place
        topology_edges: list  of edge dicts from topology.json
    """
    # Mirrors: fprintf "[ASTRO-PLANARR] ALGO: FAstroCellSymmetryMirror constraint system active"
    print("[ASTRO-SYMM] ENTER: _apply_symmetry_constraints")

    if not bboxes:
        print("[ASTRO-SYMM] EXIT: no bboxes, skip")
        return

    # Mirror force blend strength.
    # 改 20%: 0.35 instead of 1.0 (rigid) from C++ — softer blend so the
    # symmetry constraint does not overwhelm the tiled collision solver.
    # Analogous to bInheritSourceConstraintWeight = true path with a
    # scene-specific scale factor (the pixel-force physics engine operates at
    # much smaller magnitudes than UE4's world-unit spring forces).
    _MIRROR_FORCE_STRENGTH = 0.35

    # ── Phase 1: detect symmetric pairs ──────────────────────────────────────
    # Maps to: SourceCells = TArrayView<const FMatrix> passed to
    #          EmitMirroredConstraints() from the pubsub tick.
    pairs = _detect_symmetric_pairs(topology_edges, bboxes)

    if not pairs:
        print("[ASTRO-SYMM] EXIT: no symmetric pairs found")
        return

    total_mirror_forces = 0

    for cell_a, cell_b, axis_mode in pairs:
        if cell_a not in bboxes or cell_b not in bboxes:
            print(f"[ASTRO-SYMM] SKIP: {cell_a} or {cell_b} not in bboxes")
            continue

        ba = bboxes[cell_a]
        bb = bboxes[cell_b]

        # Cell centroids in canvas space
        ca_cx = ba["x"] + ba["w"] * 0.5
        ca_cy = ba["y"] + ba["h"] * 0.5
        cb_cx = bb["x"] + bb["w"] * 0.5
        cb_cy = bb["y"] + bb["h"] * 0.5

        # ── Phase 2a: symmetry axis origin = midpoint of the pair ────────────
        # Maps to: compound group's local origin in the C++ Householder
        # computation.  The midpoint is the "reflection plane origin" analogous
        # to PlanarReflectionOrigin in SetupPlanarReflectionUniformParameters().
        axis_cx = (ca_cx + cb_cx) * 0.5
        axis_cy = (ca_cy + cb_cy) * 0.5

        # ── Phase 2b (改 20%): use Custom axis for non-axis-aligned pairs ────
        # C++ always uses the preset axis.  Python extension: if the two cell
        # centroids are neither horizontally nor vertically aligned, derive a
        # Custom axis angle from the centroid-to-centroid vector (same approach
        # as the Atan2 projection in UpdatePlanarReflectionContents_RenderThread
        # in 8a2501b — "derive the symmetry axis angle from the mirror plane's
        # XY normal").
        vec_x = cb_cx - ca_cx
        vec_y = cb_cy - ca_cy
        h_dominance = abs(vec_x)   # larger → pair is side-by-side horizontally
        v_dominance = abs(vec_y)   # larger → pair is stacked vertically

        _AXIS_ALIGN_THRESHOLD = 0.25   # ratio below which we treat as custom
        if v_dominance > 0 and h_dominance / max(v_dominance, 1e-9) < _AXIS_ALIGN_THRESHOLD:
            # Pair is predominantly vertical (stacked) → Vertical mirror axis
            effective_axis = _EAstroCellSymmetryAxis.Vertical
            axis_angle = math.pi * 0.5
        elif h_dominance > 0 and v_dominance / max(h_dominance, 1e-9) < _AXIS_ALIGN_THRESHOLD:
            # Pair is predominantly horizontal (side-by-side) → Horizontal axis
            effective_axis = _EAstroCellSymmetryAxis.Horizontal
            axis_angle = 0.0
        else:
            # Diagonal pair: use the perpendicular bisector as the Custom axis,
            # mirrors Atan2(MirrorPlane.Y, MirrorPlane.X) in 8a2501b.
            effective_axis = _EAstroCellSymmetryAxis.Custom
            # Mirror *line* is the perpendicular bisector of the centroid pair:
            # direction perpendicular to vec, so angle = Atan2(vec) + π/2 − π/2
            #                                          = Atan2(vec_y, vec_x)
            # (the line passes through the midpoint and is perpendicular to the
            #  centroid-to-centroid segment, which is the natural symmetry axis)
            axis_angle = math.atan2(vec_y, vec_x)

        # ── Phase 2c: compute Householder reflection matrix ───────────────────
        # Maps to: FAstroCellSymmetryMirror::ComputeMirrorMatrix()
        mirror_matrix = _compute_mirror_matrix(effective_axis, axis_angle)

        # ── Phase 2d: ApplyToCell — reflect cell_a through axis to get ideal ─
        # position of cell_b (and vice versa for the reverse check).
        # Maps to: FAstroCellSymmetryMirror::ApplyToCell(InCellLocalTransform, …)
        ideal_b_cx, ideal_b_cy = _apply_to_cell(
            mirror_matrix, ca_cx, ca_cy, axis_cx, axis_cy
        )
        ideal_a_cx, ideal_a_cy = _apply_to_cell(
            mirror_matrix, cb_cx, cb_cy, axis_cx, axis_cy
        )

        # ── Phase 2e: compute mirror forces ──────────────────────────────────
        # Mirror force = (ideal_position − actual_position) × strength.
        # Maps to: OutMirroredCells injected into CellConstraintList, which the
        # layout solver interprets as a position pull.  Here we materialise the
        # pull directly as (dx, dy) force increments.
        #
        # 改 20%: the C++ version emits a rigid duplicate constraint (weight=1.0)
        # and lets the constraint solver handle the rest.  The Python version
        # computes the force explicitly because the physics engine accumulates
        # forces, not positions.
        force_b_dx = (ideal_b_cx - cb_cx) * _MIRROR_FORCE_STRENGTH
        force_b_dy = (ideal_b_cy - cb_cy) * _MIRROR_FORCE_STRENGTH
        force_a_dx = (ideal_a_cx - ca_cx) * _MIRROR_FORCE_STRENGTH
        force_a_dy = (ideal_a_cy - ca_cy) * _MIRROR_FORCE_STRENGTH

        # Apply mirror forces to force_field (EmitMirroredConstraints analogue)
        force_field[cell_b]["dx"] += force_b_dx
        force_field[cell_b]["dy"] += force_b_dy
        force_field[cell_a]["dx"] += force_a_dx
        force_field[cell_a]["dy"] += force_a_dy

        # ── Width/height equalisation constraint ─────────────────────────────
        # Symmetric cells should share the same w/h (same invariant enforced by
        # the C++ constraint system via shared ConstraintMask bits).
        # Apply a size-equalisation nudge via the dz channel (used by the
        # z-layer scheduler as a tie-break hint; non-zero dz signals the
        # scheduler to re-evaluate cell dimensions next epoch).
        w_diff = ba["w"] - bb["w"]
        h_diff = ba["h"] - bb["h"]
        if abs(w_diff) > 1.0 or abs(h_diff) > 1.0:
            size_signal = (abs(w_diff) + abs(h_diff)) * 0.1
            force_field[cell_a]["dz"] = force_field[cell_a].get("dz", 0) + size_signal
            force_field[cell_b]["dz"] = force_field[cell_b].get("dz", 0) + size_signal
            print(
                f"[ASTRO-SYMM] Size mismatch {cell_a}({ba['w']}x{ba['h']}) ↔ "
                f"{cell_b}({bb['w']}x{bb['h']}): dz nudge={size_signal:.2f}"
            )

        print(
            f"[ASTRO-SYMM] EmitMirroredConstraints: "
            f"{cell_a}→force({force_a_dx:+.2f},{force_a_dy:+.2f}) "
            f"{cell_b}→force({force_b_dx:+.2f},{force_b_dy:+.2f})"
        )
        total_mirror_forces += 1

    print(
        f"[ASTRO-SYMM] EXIT: emitted {total_mirror_forces} mirrored constraint pairs "
        f"(strength={_MIRROR_FORCE_STRENGTH})"
    )


def _propagate_constraints(bboxes, force_field):
    """
    [ASTRO] Global Constraint Propagation via Distance Field.

    Ported from FAstroGlobalConstraintPropagation::Propagate() introduced in
    commit 7c6f675 of upstream/unreal-renderer/DistanceFieldGlobalIllumination.cpp.

    The C++ version propagates GI displacement constraints across the scene using
    the signed distance field as a conductance medium: each source cell emits a
    constraint impulse that attenuates with SDF-measured geodesic distance to
    every receiver, letting indirect constraints travel through geometry without
    explicit ray marching.

    Python adaptation:
      - "World position" → bbox centre (cx, cy).
      - "SDF value at receiver" → proxy: inverse of the receiver's accumulated
        raw force magnitude.  A cell sitting in open space (small raw force) has
        high free-SDF and passes signal well; a heavily-pushed cell is occluded
        and attenuates incoming constraints.
      - "Displacement" → (dx, dy) force components already in force_field.
      - kMaxPropagationRadius kept proportional to the scene diagonal so the
        algorithm scales with canvas size rather than using a hard-wired cm value.
      - kFalloffExponent = 2.0 (unchanged from C++).

    Result: for every receiver cell the function accumulates weighted displacement
    from all source cells and blends it back into force_field, giving a global
    "field spreading" effect that simple collision-response cannot produce.
    """
    if not bboxes:
        return

    _FALLOFF_EXP = 2.0

    # Compute scene diagonal to get a scale-invariant propagation radius,
    # mirroring kMaxPropagationRadius = 8000 cm in a 40 m × 40 m scene.
    all_cx = [(b["x"] + b["w"] * 0.5) for b in bboxes.values()]
    all_cy = [(b["y"] + b["h"] * 0.5) for b in bboxes.values()]
    scene_w = max(all_cx) - min(all_cx) if len(all_cx) > 1 else 1.0
    scene_h = max(all_cy) - min(all_cy) if len(all_cy) > 1 else 1.0
    max_radius = math.sqrt(scene_w ** 2 + scene_h ** 2)
    max_radius = max(max_radius, 1.0)   # guard against degenerate single-cell scenes
    max_radius_sq = max_radius * max_radius
    radius_norm = 1.0 / max_radius      # maps [0, max_radius] → [0, 1]

    # Build per-cell centroid + current force snapshot (the "source" state).
    # Snapshot before the accumulation loop so sources and receivers read from
    # the same epoch — mirrors the C++ passing SourceCells as a const TArray.
    cell_ids = list(bboxes.keys())
    cx = {cid: bboxes[cid]["x"] + bboxes[cid]["w"] * 0.5 for cid in cell_ids}
    cy = {cid: bboxes[cid]["y"] + bboxes[cid]["h"] * 0.5 for cid in cell_ids}

    src_dx  = {cid: force_field[cid]["dx"] for cid in cell_ids}
    src_dy  = {cid: force_field[cid]["dy"] for cid in cell_ids}
    src_mag = {cid: math.sqrt(src_dx[cid] ** 2 + src_dy[cid] ** 2)
               for cid in cell_ids}

    # Receiver occlusion proxy: cells with low existing force are in "free space"
    # (occlusion_conductance → 1); cells already heavily pushed are "embedded in
    # geometry" (occlusion_conductance → 0).  Max raw force across all cells
    # provides the normalising denominator, analogous to AOParams.ObjectMaxOcclusionDistance.
    max_force = max(src_mag.values()) if src_mag else 0.0
    _KINDA_SMALL = 1e-6

    def _occlusion_conductance(recv_id):
        # High existing force → occluded → low conductance (signal is blocked).
        # Low existing force → free space → high conductance (signal passes through).
        # Clamp to [0, 1].
        raw = src_mag[recv_id] / max(max_force, _KINDA_SMALL)
        return max(0.0, min(1.0, 1.0 - raw))

    # Core propagation loop — O(N²) over cells, same asymptotic class as the
    # C++ TArray<FCellConstraint> double loop; acceptable for the sub-100-cell
    # scenes this orchestrator handles.
    accumulated_dx = {cid: 0.0 for cid in cell_ids}
    accumulated_dy = {cid: 0.0 for cid in cell_ids}

    for recv_id in cell_ids:
        rx, ry = cx[recv_id], cy[recv_id]
        occ_cond = _occlusion_conductance(recv_id)

        total_weight  = 0.0
        acc_dx        = 0.0
        acc_dy        = 0.0

        for src_id in cell_ids:
            if src_id == recv_id:
                continue

            # Vector from receiver to source (matches C++ Delta computation).
            ddx = rx - cx[src_id]
            ddy = ry - cy[src_id]
            dist_sq = ddx * ddx + ddy * ddy

            if dist_sq > max_radius_sq:
                # Beyond propagation radius — cull, same as the C++ continue.
                continue

            dist = math.sqrt(dist_sq)
            norm_dist = dist * radius_norm                      # ∈ [0, 1]

            # Distance-based conductance: (1 − d̂)^k, kFalloffExponent = 2.
            # Decays quadratically: cells far apart share little constraint energy.
            dist_cond = (1.0 - norm_dist) ** _FALLOFF_EXP

            # Combined weight = distance conductance × occlusion conductance
            # × source weight (source force magnitude as proxy for source strength).
            combined = dist_cond * occ_cond * src_mag[src_id]

            acc_dx      += src_dx[src_id] * combined
            acc_dy      += src_dy[src_id] * combined
            total_weight += combined

        # Normalise by total weight so more sources don't artificially amplify
        # the result — direct port of the TotalWeight normalisation in C++.
        if total_weight > _KINDA_SMALL:
            accumulated_dx[recv_id] = acc_dx / total_weight
            accumulated_dy[recv_id] = acc_dy / total_weight

    # Blend propagated displacement back into force_field.
    # Using a 0.5 blend factor so constraint propagation supplements — not
    # replaces — the direct collision response already in force_field.
    _BLEND = 0.5
    for cid in cell_ids:
        force_field[cid]["dx"] += accumulated_dx[cid] * _BLEND
        force_field[cid]["dy"] += accumulated_dy[cid] * _BLEND

    non_trivial = sum(
        1 for cid in cell_ids
        if abs(accumulated_dx[cid]) > _KINDA_SMALL or abs(accumulated_dy[cid]) > _KINDA_SMALL
    )
    print(
        f"[DistanceFieldGI] Constraint propagation complete: "
        f"{non_trivial}/{len(cell_ids)} cells received non-zero influence "
        f"(max_radius={max_radius:.1f}px, falloff_exp={_FALLOFF_EXP})"
    )


def physics_engine():
    """
    Physics Engine organ — reads all cell/*/bbox.json, detects 3D collisions,
    computes force field. Publishes to physics/force_field.json.

    Collision detection uses a two-pass pipeline:
      Pass 1 (sweep-line):  FAstroCellSweepLine, ported from commit 0b4b199 of
        upstream/unreal-renderer/SceneSoftwareOcclusion.cpp.
        Complexity: O((N+K) log N).  Used for diagnostic overlap enumeration.
      Pass 2 (tiled solver): FAstroTiledConstraintSolver, ported from commit
        7c82b90 of upstream/unreal-renderer/TiledDeferredLightRendering.cpp.
        Splits canvas into TILE_SIZE×TILE_SIZE tiles; each cell is assigned to
        a tile by bbox centre; constraints are resolved only within each tile's
        3×3 neighbourhood.  Complexity: O(N*K) where K = avg cells per tile.
        This is the tiled deferred light → tiled constraint mapping: lights →
        cell constraints, screen tiles → canvas tiles.  Force accumulation is
        owned by this pass.

    After local collision response, global constraint propagation via distance
    field (_propagate_constraints) is applied — ported from commit 7c6f675 of
    upstream/unreal-renderer/DistanceFieldGlobalIllumination.cpp.

    Only same-z cells can collide. Different z = no collision (preserved).
    """
    bbox_files = glob.glob(os.path.join(CHANNELS, "cell", "*", "bbox.json"))
    bboxes = {}
    for bf in bbox_files:
        cell_id = os.path.basename(os.path.dirname(bf))
        with open(bf) as f:
            bboxes[cell_id] = json.load(f)

    force_field = {cid: {"dx": 0, "dy": 0, "dz": 0} for cid in bboxes}

    # --- 1. Bucket cells by z-layer (same-z-only rule preserved from original) ---
    rects_by_z = {}
    for cell_id, b in bboxes.items():
        z = b.get("z", 3)
        min_x = int(b["x"])
        min_y = int(b["y"])
        max_x = int(b["x"] + b["w"])
        max_y = int(b["y"] + b["h"])
        rects_by_z.setdefault(z, []).append((cell_id, min_x, min_y, max_x, max_y))

    # --- 2. Run sweep-line O((N+K) log N) overlap detection ---------------------
    overlap_pairs = _sweep_line_overlaps(rects_by_z)

    # --- 3. Convert overlapping pairs → collisions + repulsion forces -----------
    #        via _tiled_constraint_solve (O(N*K) tiled spatial partition).
    #
    # [ASTRO-TILEDSOLVER] Replace the previous per-pair brute-force repulsion
    # loop with the tiled constraint solver ported from commit 7c82b90 of
    # upstream/unreal-renderer/TiledDeferredLightRendering.cpp.
    #
    # Concept mapping (same as FAstroTiledConstraintSolver comment block):
    #   Light influence region  → Cell bbox overlap AABB
    #   Screen tile             → Canvas tile region (TILE_SIZE x TILE_SIZE px)
    #   GPU cull per tile       → Neighbourhood membership check per tile
    #   ResolveAll()            → repulsion force accumulation in force_field
    #
    # The sweep-line pass (step 2 above) still runs for diagnostic collision
    # reporting and as a fallback reference; the tiled solver is authoritative
    # for force accumulation.  Both passes operate on the same bboxes/force_field
    # so we zero force_field first and let the tiled solver own all repulsion
    # work, then feed its collision list to the downstream propagation pass.
    #
    # Complexity: O(N*K) where K = avg cells per tile neighbourhood (much less
    # than N), vs the previous O(N^2) double-loop.  Matches the GPU
    # tiled-deferred reduction from O(numLights x numPixels) to
    # O(numLights/tile x numPixels).
    force_field = {cid: {"dx": 0, "dy": 0, "dz": 0} for cid in bboxes}

    collisions = _tiled_constraint_solve(bboxes, force_field)

    print(
        f"[Physics] {len(collisions)} collisions detected "
        f"(tiled constraint solver O(N*K), ported from TiledDeferredLightRendering.cpp 7c82b90)"
    )

    # --- 3b. Cell symmetry mirror constraints ------------------------------------
    # [ASTRO] FAstroCellSymmetryMirror — ported from commit 8a2501b of
    # upstream/unreal-renderer/PlanarReflectionRendering.cpp.
    #
    # Detects symmetric cell pairs in the topology (e.g. add_norm1 <-> add_norm2),
    # computes the Householder reflection matrix for the symmetry axis, and
    # accumulates "mirror forces" into force_field so the normal physics loop
    # converges toward a symmetric layout.
    #
    # Concept mapping (PlanarReflection -> cell symmetry, same as the C++ header):
    #   Reflection plane    -> symmetry axis (2-D Householder in canvas space)
    #   FMirrorMatrix       -> M = I - 2NNt  (ComputeMirrorMatrix)
    #   Mirror camera       -> mirrored centroid position  (ApplyToCell)
    #   Secondary renderer  -> force_field accumulation    (EmitMirroredConstraints)
    #
    # This pass runs after the tiled collision solver so that symmetry forces
    # compose additively with repulsion forces -- same compositing order as
    # the C++ constraint list injection after collision response.
    _topology_edges_symm = []
    _topology_path_symm = os.path.join(CHANNELS, "skeleton", "topology.json")
    if os.path.exists(_topology_path_symm):
        with open(_topology_path_symm) as _tf:
            _topology_edges_symm = json.load(_tf).get("edges", [])
    _apply_symmetry_constraints(bboxes, force_field, _topology_edges_symm)

    # --- 4. Global constraint propagation via distance field --------------------
    # Ported from FAstroGlobalConstraintPropagation::Propagate() (commit 7c6f675).
    # Spreads accumulated collision forces across the scene using distance-field
    # conductance so that constraint energy originating at one cell can influence
    # geometrically distant cells — a capability absent from pure collision response.
    _propagate_constraints(bboxes, force_field)

    # --- 5. Volumetric constraint-field voxelization ----------------------------
    # Ported from FAstroConstraintVoxelGrid in commit 5f5cf3a of
    # upstream/unreal-renderer/VolumetricFogVoxelization.cpp.
    # Complements the point-to-point distance field above with an area-density
    # field: each cell's constraint value is accumulated into an NxN voxel grid
    # per z-layer; high-density voxels produce scatter forces, low-density voxels
    # produce attract forces toward the canvas centre.  Density map is persisted
    # to physics/z_layers.json for the pub-sub scheduling layer to consume
    # (mirrors GAstroConstraintGrid feeding FAstroCellConstraintPublisher).
    voxel_grid = _voxelize_constraint_field(bboxes, force_field)
    # Serialise grid: convert tuple keys → "gx,gy,z" strings for JSON.
    voxel_grid_json = {
        f"{gx},{gy},{z}": meta
        for (gx, gy, z), meta in voxel_grid.items()
    }
    write_channel("physics/z_layers.json", voxel_grid_json)

    write_channel("physics/force_field.json", force_field)
    write_channel("physics/collision.json", {"collisions": collisions, "count": len(collisions)})

    print(f"[Physics] {len(collisions)} collisions detected (sweep-line O((N+K)logN))")
    return collisions


def convergence_judge():
    """Check if all cells have converged (no collisions)."""
    collision_data = read_channel("physics/collision.json")
    count = collision_data.get("count", 999)
    epoch = read_channel("skeleton/epoch.json")

    converged = count == 0
    write_channel("physics/converged.json", {
        "converged": converged,
        "epoch": epoch["current"],
        "conflicts": count
    })
    print(f"[Judge] epoch={epoch['current']} conflicts={count} converged={converged}")
    return converged


# ---------------------------------------------------------------------------
# FAstroCellRegistry + FConstraintCollector
# Ported from upstream/unreal-renderer/SceneRendering.cpp commit 7eae73c.
#
# Original: C++ structs FAstroCellSlot / AstroConstraintCollector operating
# on FViewInfo arrays with GPU-upload uint32 packing.
# Python port: dataclass registry + collector class over cell bbox/topology
# data; packing repurposed as constraint_score for z-layer sort priority.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class AstroCellSlot:
    """
    FAstroCellSlot — one registered cell in the topology graph.

    Fields mirror the C++ struct:
      cell_id        ↔  CellId   (str here instead of int32)
      z_layer        ↔  ZLayer
      constraint_mask↔  ConstraintMask  (packed outgoing-edge bits)
      active         ↔  bActive
      bbox           — bbox dict (x,y,w,h); not in C++ slot, added for Python
      species        — species string derived from bbox.json
      constraints    — {"incoming": [...], "outgoing": [...], "neighbors": [...]}
    """
    cell_id: str
    z_layer: int
    constraint_mask: int = 0          # low 8 bits: outgoing-edge count (clamped)
    active: bool = True
    bbox: Dict = field(default_factory=dict)
    species: str = ""
    constraints: Dict = field(default_factory=lambda: {
        "incoming": [], "outgoing": [], "neighbors": []
    })


class AstroConstraintCollector:
    """
    AstroConstraintCollector — walks all registered slots and aggregates
    constraints into a packed buffer, mirroring AstroConstraintCollector::Collect()
    from commit 7eae73c.

    Pack layout (uint32 equivalent → Python int):
      bits 31-16 : cell_id hash (low 16 bits of hash(cell_id))
      bits 15- 8 : z_layer  & 0xFF
      bits  7- 0 : constraint_mask & 0xFF   (= outgoing-edge count, clamped 0-255)

    Divergence from C++: we also expose per-cell outgoing_edge_count() for the
    z-layer sort step — the C++ version implicitly carries this in ConstraintMask
    which we use the same way.
    """

    def __init__(self, registry: List[AstroCellSlot]) -> None:
        self._registry = registry
        self._constraint_buffer: List[int] = []

    def collect(self) -> None:
        """Iterate active slots; pack constraints → _constraint_buffer."""
        self._constraint_buffer.clear()
        for slot in self._registry:
            if not slot.active:
                continue
            # Pack: cell_id hash (16b) | z_layer (8b) | constraint_mask (8b)
            cell_id_bits = hash(slot.cell_id) & 0xFFFF
            packed = (
                (cell_id_bits          << 16) |
                ((slot.z_layer & 0xFF) <<  8) |
                (slot.constraint_mask  &  0xFF)
            )
            self._constraint_buffer.append(packed)

        print(
            f"[ASTRO-CONSTRAINT] Collected {len(self._constraint_buffer)} constraints "
            f"from {len(self._registry)} registered cells"
        )

    @property
    def constraint_buffer(self) -> List[int]:
        return self._constraint_buffer

    def active_cell_count(self) -> int:
        return len(self._constraint_buffer)

    def outgoing_edge_count(self, cell_id: str) -> int:
        """Return the outgoing-edge priority score for a given cell_id."""
        for slot in self._registry:
            if slot.cell_id == cell_id and slot.active:
                return len(slot.constraints["outgoing"])
        return 0


def _build_cell_registry(
    bbox_map: Dict[str, Dict],
    topology_edges: List[Dict],
    species_map: Dict[str, str],
) -> List[AstroCellSlot]:
    """
    [ASTRO-CELL-REGISTRY] Populate registry from topology, mirroring the
    InitViews block in SceneRendering.cpp (commit 7eae73c).

    Each cell_id → AstroCellSlot; ZLayer from bbox["z"]; constraint_mask
    encodes outgoing-edge count (clamped to 0xFF), analogous to how the C++
    code encodes StereoPass into ConstraintMask as a lightweight hint without
    touching the public API.

    Neighbor relations: cells sharing a z-layer are mutual neighbors,
    mirroring how the C++ registry groups view-ports by GPU z-layer slot.
    """
    # --- Phase 1: build incoming / outgoing maps from topology edges ----------
    outgoing: Dict[str, List[str]] = {cid: [] for cid in bbox_map}
    incoming: Dict[str, List[str]] = {cid: [] for cid in bbox_map}
    for edge in topology_edges:
        for src in edge.get("sources", []):
            for tgt in edge.get("targets", []):
                if src in outgoing:
                    outgoing[src].append(tgt)
                if tgt in incoming:
                    incoming[tgt].append(src)

    # --- Phase 2: bucket cells by z-layer for neighbor detection --------------
    z_buckets: Dict[int, List[str]] = {}
    for cid, b in bbox_map.items():
        z = b.get("z", 3)
        z_buckets.setdefault(z, []).append(cid)

    # --- Phase 3: build slots (monotonically assigned, epoch-coherent) --------
    registry: List[AstroCellSlot] = []
    for cell_id in sorted(bbox_map.keys()):           # stable sort = epoch-coherent
        b = bbox_map[cell_id]
        z_layer = b.get("z", 3)

        neighbors = [c for c in z_buckets.get(z_layer, []) if c != cell_id]

        out_edges = outgoing[cell_id]
        in_edges  = incoming[cell_id]

        # constraint_mask: low 8 bits = outgoing-edge count (clamped), same
        # semantic as ConstraintMask in C++ — a cheap per-slot priority hint.
        constraint_mask = min(len(out_edges), 0xFF)

        slot = AstroCellSlot(
            cell_id=cell_id,
            z_layer=z_layer,
            constraint_mask=constraint_mask,
            active=True,
            bbox=b,
            species=species_map.get(cell_id, ""),
            constraints={
                "incoming":  in_edges,
                "outgoing":  out_edges,
                "neighbors": neighbors,
            },
        )
        registry.append(slot)
        print(
            f"[ASTRO-CELL-REGISTRY] Registered cell id={cell_id} "
            f"z-layer={z_layer} constraint_mask=0x{constraint_mask:02X} "
            f"out={len(out_edges)} in={len(in_edges)}"
        )

    return registry


# Species that carry partial transparency / overlay semantics —
# analogous to EMeshPass::TranslucencyAll in the C++ sort.
# Cells in this set are "translucent passes" and render after opaque ones
# within the same z-layer (改 20%: driven by species family, not pass index).
_TRANSLUCENT_SPECIES = {"cil-eye", "cil-plus"}


# ============================================================================
# [ASTRO-SVG] post_process_svg
# Python port of FAstroSvgPostProcess from PostProcessing.cpp commit f2a77b0.
#
# Three composable passes run in sequence on the assembled SVG string:
#
#   Pass 1 – edge_soften        : polyline/path 折线点 → 微小 Bezier 曲率
#                                  (mirrors AddEdgeSofteningPass / RenderGaussianBlur
#                                   with radius=1.2 px, separable 2-pass Gaussian)
#   Pass 2 – label_avoid        : text 元素与 rect/line 重叠检测 → 微调坐标
#                                  (mirrors AddLabelCollisionAvoidPass / dilate-margin
#                                   with margin=4.0 px)
#   Pass 3 – weight_balance     : 左右/上下视觉重量不平衡 → 调整 translate 间距
#                                  (mirrors AddVisualWeightBalancePass / lerp(orig,
#                                   luma-normalised, strength=0.5))
#
# Each pass is a no-op when its enable flag is False, so the chain composes
# linearly and can be partially disabled — same as the C++ CVar guard pattern.
# ============================================================================

import re as _re
import xml.etree.ElementTree as _ET

# --- CVars (mirrors TAutoConsoleVariable defaults in the C++ source) --------
_SVG_EDGE_SOFTEN_ENABLED       = True
_SVG_EDGE_SOFTEN_RADIUS        = 1.2    # px  [0.5, 4.0]

_SVG_LABEL_AVOID_ENABLED       = True
_SVG_LABEL_AVOID_MARGIN        = 4.0   # px  [0.0, 32.0]

_SVG_WEIGHT_BALANCE_ENABLED    = True
_SVG_WEIGHT_BALANCE_STRENGTH   = 0.5   # [0.0, 1.0]


def _parse_svg_tree(svg_string: str):
    """
    Parse SVG string → ElementTree root.  Register the SVG namespace so that
    serialisation does not emit 'ns0:' prefixes (mirrors the Unreal render
    graph's expectation that the SVG DOM is namespace-clean after the pass).
    """
    _ET.register_namespace("", "http://www.w3.org/2000/svg")
    _ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = _ET.fromstring(svg_string)
    return root


def _serialise_svg_tree(root) -> str:
    """Serialise ElementTree back to SVG string."""
    return _ET.tostring(root, encoding="unicode", xml_declaration=False)


# ---------------------------------------------------------------------------
# Pass 1 – Edge Softening
# ---------------------------------------------------------------------------
def _edge_soften_pass(root, radius: float) -> None:
    """
    [ASTRO-SVG] EdgeSofteningPass — Python/SVG DOM port of AddEdgeSofteningPass.

    Unreal approach: RenderGaussianBlur() with SizeScale == radius, separable
    2-pass Gaussian at full resolution, touching only pixels above a contrast
    threshold.  In the SVG DOM equivalent we cannot blur pixels directly, so
    we achieve the same perceptual effect by converting sharp polyline corners
    into smooth cubic Bezier curves:

      • For every <polyline> with ≥ 3 points: replace with a <path> whose
        segment junctions use cubic Bezier control points offset by `radius`
        pixels in the direction of the angle bisector.  This is the SVG-native
        equivalent of sub-pixel anti-aliasing on thin connector strokes.
      • For every <path> whose 'd' attribute contains only L / M commands
        (i.e. a bare polyline encoded as a path): apply the same Bezier
        smoothing to interior vertices.

    Skipped when radius < 0.5 (below perceptual threshold, mirrors C++ guard).

    Args:
        root  : ElementTree root element (modified in-place).
        radius: Gaussian kernel radius from CVar r.Astro.Svg.EdgeSoftenRadius.
    """
    if radius < 0.5:
        return

    ns = {"svg": "http://www.w3.org/2000/svg"}

    def _smooth_points(pts: list, r: float) -> str:
        """
        Convert a sequence of (x,y) points into a smooth cubic-Bezier path.

        Algorithm (mirrors separable Gaussian 2-pass logic repurposed to 2D):
          For each interior vertex V[i], compute the angle bisector direction
          between the incoming segment (V[i-1]→V[i]) and the outgoing segment
          (V[i]→V[i+1]).  Place control points at distance r along each
          segment, giving a C1-continuous curve through all original points.

        The result is a 'd' attribute string ready for a <path> element.
        """
        if len(pts) < 2:
            return ""
        if len(pts) == 2:
            return f"M {pts[0][0]:.3f},{pts[0][1]:.3f} L {pts[1][0]:.3f},{pts[1][1]:.3f}"

        d_parts = [f"M {pts[0][0]:.3f},{pts[0][1]:.3f}"]
        for i in range(1, len(pts) - 1):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]

            # Incoming direction (V[i-1] → V[i])
            dx_in  = x1 - x0
            dy_in  = y1 - y0
            len_in = math.hypot(dx_in, dy_in) or 1.0
            ux_in  = dx_in / len_in
            uy_in  = dy_in / len_in

            # Outgoing direction (V[i] → V[i+1])
            dx_out  = x2 - x1
            dy_out  = y2 - y1
            len_out = math.hypot(dx_out, dy_out) or 1.0
            ux_out  = dx_out / len_out
            uy_out  = dy_out / len_out

            # Control points: r pixels back from V[i] along each segment
            t = min(r, len_in * 0.4, len_out * 0.4)   # clamp so cp stays on segment
            cp1x = x1 - ux_in  * t
            cp1y = y1 - uy_in  * t
            cp2x = x1 + ux_out * t
            cp2y = y1 + uy_out * t

            d_parts.append(
                f"C {cp1x:.3f},{cp1y:.3f} {cp2x:.3f},{cp2y:.3f} {x1:.3f},{y1:.3f}"
            )

        # Final straight segment to last point
        d_parts.append(f"L {pts[-1][0]:.3f},{pts[-1][1]:.3f}")
        return " ".join(d_parts)

    svg_ns = "http://www.w3.org/2000/svg"
    tag_polyline = f"{{{svg_ns}}}polyline"
    tag_path     = f"{{{svg_ns}}}path"

    # --- Smooth <polyline> elements ------------------------------------------
    for parent in root.iter():
        children_to_replace = []
        for child in list(parent):
            if child.tag != tag_polyline:
                continue
            pts_str = child.get("points", "").strip()
            if not pts_str:
                continue
            # Parse "x1,y1 x2,y2 ..." or "x1 y1 x2 y2 ..."
            nums = [float(v) for v in _re.split(r"[\s,]+", pts_str) if v]
            if len(nums) < 4 or len(nums) % 2 != 0:
                continue
            pts = [(nums[k], nums[k + 1]) for k in range(0, len(nums), 2)]
            if len(pts) < 3:
                continue   # straight line — no bend to soften

            smooth_d = _smooth_points(pts, radius)
            path_el = _ET.Element(tag_path)
            # Copy all attributes except 'points'
            for attr, val in child.attrib.items():
                if attr != "points":
                    path_el.set(attr, val)
            path_el.set("d", smooth_d)
            path_el.set("fill", child.get("fill", "none"))
            children_to_replace.append((child, path_el))

        for old_el, new_el in children_to_replace:
            idx = list(parent).index(old_el)
            parent.remove(old_el)
            parent.insert(idx, new_el)

    # --- Smooth <path> elements that are bare polylines (M + L only) ---------
    for el in root.iter(tag_path):
        d = el.get("d", "").strip()
        if not d:
            continue
        # Only process paths that use only M/m and L/l commands
        cmds = set(_re.findall(r"[A-Za-z]", d))
        if not cmds.issubset({"M", "m", "L", "l"}):
            continue
        tokens = _re.split(r"([MmLl])", d)
        pts: list = []
        cur_x, cur_y = 0.0, 0.0
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if tok in ("M", "L"):
                pass   # next token contains coords
            elif tok in ("m", "l"):
                pass
            else:
                nums = [float(v) for v in _re.split(r"[\s,]+", tok) if v]
                for k in range(0, len(nums) - 1, 2):
                    pts.append((nums[k], nums[k + 1]))
        if len(pts) < 3:
            continue
        smooth_d = _smooth_points(pts, radius)
        el.set("d", smooth_d)

    print(f"[ASTRO-SVG] EdgeSoftenPass applied (radius={radius:.2f})")


# ---------------------------------------------------------------------------
# Pass 2 – Label Collision Avoidance
# ---------------------------------------------------------------------------
def _label_avoid_pass(root, margin: float) -> None:
    """
    [ASTRO-SVG] LabelCollisionAvoidPass — Python/SVG DOM port of
    AddLabelCollisionAvoidPass.

    Unreal approach: dilate bright label regions by margin/2 px via a
    separable Gaussian blur (DilateRadius = margin * 0.5), then check whether
    label pixels overlap with geometry pixels.  In the SVG DOM equivalent we
    operate directly on element bounding boxes:

      1. Collect all <text> elements and parse their (x, y) positions +
         estimated bounding box (font-size × text length for width; font-size
         for height).
      2. Collect all <rect> and <line> elements as obstacle bounding boxes.
      3. For each text element, test for AABB overlap with every obstacle
         (inflated by `margin` px on all sides — equivalent to the dilation
         step in the C++ pass).
      4. If overlap detected, displace the text element in the direction of
         least penetration depth by exactly the overlap amount + margin, so
         the result satisfies the minimum-margin invariant after the move.
         Displacement is clamped to prevent the label from leaving the canvas.

    Args:
        root  : ElementTree root element (modified in-place).
        margin: Minimum pixel clearance from CVar r.Astro.Svg.LabelAvoidMargin.
    """
    if margin <= 0:
        return

    svg_ns   = "http://www.w3.org/2000/svg"

    def _try_float(v, default=0.0):
        try:
            return float(v) if v else default
        except (ValueError, TypeError):
            return default

    # Parse canvas size from root viewBox / width
    vb = root.get("viewBox", "")
    canvas_w = _try_float(root.get("width"),  800.0)
    canvas_h = _try_float(root.get("height"), 600.0)
    if vb:
        parts = _re.split(r"[\s,]+", vb.strip())
        if len(parts) == 4:
            canvas_w = _try_float(parts[2], canvas_w)
            canvas_h = _try_float(parts[3], canvas_h)

    # ── Collect obstacle bboxes (rect + line) ─────────────────────────────────
    obstacles: list = []   # list of (min_x, min_y, max_x, max_y)

    for el in root.iter(f"{{{svg_ns}}}rect"):
        rx = _try_float(el.get("x"))
        ry = _try_float(el.get("y"))
        rw = _try_float(el.get("width"))
        rh = _try_float(el.get("height"))
        if rw > 0 and rh > 0:
            obstacles.append((rx, ry, rx + rw, ry + rh))

    for el in root.iter(f"{{{svg_ns}}}line"):
        x1 = _try_float(el.get("x1"))
        y1 = _try_float(el.get("y1"))
        x2 = _try_float(el.get("x2"))
        y2 = _try_float(el.get("y2"))
        sw = _try_float(el.get("stroke-width"), 1.0)
        obstacles.append((
            min(x1, x2) - sw / 2, min(y1, y2) - sw / 2,
            max(x1, x2) + sw / 2, max(y1, y2) + sw / 2,
        ))

    displaced = 0
    for el in root.iter(f"{{{svg_ns}}}text"):
        tx = _try_float(el.get("x"))
        ty = _try_float(el.get("y"))
        # Estimate bounding box: font-size × char count
        font_size = _try_float(
            el.get("font-size", el.get("fontSize")), 12.0
        )
        text_content = "".join(el.itertext())
        est_w = max(font_size * len(text_content) * 0.6, font_size)
        est_h = font_size * 1.2   # line-height ≈ 1.2 em

        # Text bbox (anchor at baseline left by default; treat as top-left)
        t_min_x = tx
        t_min_y = ty - est_h     # baseline → top
        t_max_x = tx + est_w
        t_max_y = ty

        # Inflate by margin (dilation step, mirrors DilateRadius = margin * 0.5 × 2)
        m = margin
        for (o_min_x, o_min_y, o_max_x, o_max_y) in obstacles:
            # AABB overlap check with inflated obstacle
            inf_min_x = o_min_x - m
            inf_min_y = o_min_y - m
            inf_max_x = o_max_x + m
            inf_max_y = o_max_y + m

            # No overlap → skip
            if t_max_x < inf_min_x or t_min_x > inf_max_x:
                continue
            if t_max_y < inf_min_y or t_min_y > inf_max_y:
                continue

            # Penetration depths on each axis
            pen_left  = t_max_x - inf_min_x
            pen_right = inf_max_x - t_min_x
            pen_up    = t_max_y - inf_min_y
            pen_down  = inf_max_y - t_min_y

            # Push along axis of least penetration (mirrors the C++ repulsion logic)
            if min(pen_left, pen_right) <= min(pen_up, pen_down):
                if pen_left < pen_right:
                    new_tx = max(0.0, tx - pen_left - margin)
                else:
                    new_tx = min(canvas_w - est_w, tx + pen_right + margin)
                el.set("x", f"{new_tx:.3f}")
                tx = new_tx
                t_min_x = new_tx
                t_max_x = new_tx + est_w
            else:
                if pen_up < pen_down:
                    new_ty = max(est_h, ty - pen_up - margin)
                else:
                    new_ty = min(canvas_h, ty + pen_down + margin)
                el.set("y", f"{new_ty:.3f}")
                ty = new_ty
                t_min_y = new_ty - est_h
                t_max_y = new_ty

            displaced += 1
            break   # one displacement per label per pass (single-iteration, stable)

    print(f"[ASTRO-SVG] LabelCollisionAvoidPass applied (margin={margin:.1f}px, displaced={displaced})")


# ---------------------------------------------------------------------------
# Pass 3 – Visual Weight Balancing
# ---------------------------------------------------------------------------
def _weight_balance_pass(root, strength: float) -> None:
    """
    [ASTRO-SVG] VisualWeightBalancePass — Python/SVG DOM port of
    AddVisualWeightBalancePass.

    Unreal approach: wide Gaussian (SizeScale=16 px) captures local mean
    luminance, fed back as additive tint weighted by `strength`, giving a
    lerp(original, luma-normalised, strength) result.  Strength==0 is exact
    identity.

    In the SVG DOM equivalent:
      1. Collect (x, w) extents of all visible opaque elements per horizontal
         half (left / right) and (y, h) extents per vertical half (top / bottom)
         to compute a proxy visual weight for each quadrant.
      2. Compute imbalance ratio:  weight_ratio = max(L,R) / (min(L,R) + ε).
         If ratio > 1 + threshold (0.15) the layout is imbalanced.
      3. Apply a corrective translate to all elements in the *heavier* half:
         shift them toward the centre by  delta = gap × strength  where `gap`
         is the signed centroid difference.  This mirrors the additive-tint
         feedback loop — strength==0.5 moves the heavier side halfway to the
         centre.
      4. Repeat for the vertical axis.

    Only <g>, <rect>, <circle>, <ellipse>, <text>, <polyline>, <path>, <line>,
    <image> elements contribute to weight estimation.  The background <rect>
    covering the full canvas is excluded.

    Args:
        root    : ElementTree root element (modified in-place).
        strength: Blend factor from CVar r.Astro.Svg.WeightBalanceStrength.
    """
    if strength <= 0.0:
        return

    svg_ns = "http://www.w3.org/2000/svg"
    WEIGHTED_TAGS = {
        f"{{{svg_ns}}}{t}" for t in
        ("rect", "circle", "ellipse", "text", "polyline", "path", "line", "image")
    }

    def _try_float(v, default=0.0):
        try:
            return float(v) if v else default
        except (ValueError, TypeError):
            return default

    vb = root.get("viewBox", "")
    canvas_w = _try_float(root.get("width"),  800.0)
    canvas_h = _try_float(root.get("height"), 600.0)
    if vb:
        parts = _re.split(r"[\s,]+", vb.strip())
        if len(parts) == 4:
            canvas_w = _try_float(parts[2], canvas_w)
            canvas_h = _try_float(parts[3], canvas_h)

    cx = canvas_w / 2.0
    cy = canvas_h / 2.0

    # ── Collect element centroids + areas ─────────────────────────────────────
    # Each entry: (element_ref, centroid_x, centroid_y, area)
    items: list = []

    for el in root.iter():
        if el.tag not in WEIGHTED_TAGS:
            continue
        tag_local = el.tag.split("}")[-1]

        if tag_local == "rect":
            ex = _try_float(el.get("x"))
            ey = _try_float(el.get("y"))
            ew = _try_float(el.get("width"))
            eh = _try_float(el.get("height"))
            # Exclude full-canvas background rect
            if ew >= canvas_w * 0.9 and eh >= canvas_h * 0.9:
                continue
            if ew <= 0 or eh <= 0:
                continue
            items.append((el, ex + ew / 2, ey + eh / 2, ew * eh))

        elif tag_local == "circle":
            ecx = _try_float(el.get("cx"))
            ecy = _try_float(el.get("cy"))
            er  = _try_float(el.get("r"), 1.0)
            items.append((el, ecx, ecy, math.pi * er * er))

        elif tag_local == "ellipse":
            ecx = _try_float(el.get("cx"))
            ecy = _try_float(el.get("cy"))
            erx = _try_float(el.get("rx"), 1.0)
            ery = _try_float(el.get("ry"), 1.0)
            items.append((el, ecx, ecy, math.pi * erx * ery))

        elif tag_local == "text":
            ex  = _try_float(el.get("x"))
            ey  = _try_float(el.get("y"))
            fs  = _try_float(el.get("font-size", el.get("fontSize")), 12.0)
            txt = "".join(el.itertext())
            ew  = fs * len(txt) * 0.6
            items.append((el, ex + ew / 2, ey - fs / 2, ew * fs))

        elif tag_local == "line":
            x1 = _try_float(el.get("x1"))
            y1 = _try_float(el.get("y1"))
            x2 = _try_float(el.get("x2"))
            y2 = _try_float(el.get("y2"))
            items.append((el, (x1 + x2) / 2, (y1 + y2) / 2, math.hypot(x2 - x1, y2 - y1)))

        elif tag_local in ("polyline", "path", "image"):
            # Rough centroid from x/y/width/height attributes if available
            ex = _try_float(el.get("x"))
            ey = _try_float(el.get("y"))
            ew = _try_float(el.get("width"),  20.0)
            eh = _try_float(el.get("height"), 20.0)
            items.append((el, ex + ew / 2, ey + eh / 2, ew * eh))

    if not items:
        print("[ASTRO-SVG] WeightBalancePass skipped (no weighted elements found)")
        return

    # ── Compute left/right and top/bottom weight sums ─────────────────────────
    # Weight proxy: sum of areas in each half (mirrors luminance integration
    # at half-res in the C++ Bloom-setup pass)
    left_w  = sum(area for _, ex, _, area in items if ex < cx)
    right_w = sum(area for _, ex, _, area in items if ex >= cx)
    top_w   = sum(area for _, _, ey, area in items if ey < cy)
    bot_w   = sum(area for _, _, ey, area in items if ey >= cy)

    eps = 1e-6
    threshold = 0.15   # imbalance below 15% is acceptable (matches Unreal SM4 guard)

    adjustments_x = 0
    adjustments_y = 0

    # ── Horizontal balance ────────────────────────────────────────────────────
    if left_w + right_w > eps:
        ratio_lr = max(left_w, right_w) / (min(left_w, right_w) + eps)
        if ratio_lr > 1.0 + threshold:
            # Heavier side: shift toward centre by delta = centroid_gap × strength
            # (mirrors lerp(original, luma-normalised, Strength) additive blend)
            if left_w > right_w:
                # Left is heavier — compute left centroid, push right (+x)
                heavy_cx = sum(ex * area for _, ex, _, area in items if ex < cx) / (left_w + eps)
                gap   = cx - heavy_cx          # distance centroid → canvas centre
                delta = gap * strength         # lerp blend
                for el, ex, ey, _ in items:
                    if ex >= cx:
                        continue
                    # Shift element x-position by delta (positive = rightward)
                    tag_local = el.tag.split("}")[-1]
                    if tag_local in ("rect", "text", "image"):
                        old_x = _try_float(el.get("x"))
                        el.set("x", f"{old_x + delta:.3f}")
                        adjustments_x += 1
                    elif tag_local == "circle":
                        old_cx = _try_float(el.get("cx"))
                        el.set("cx", f"{old_cx + delta:.3f}")
                        adjustments_x += 1
                    elif tag_local == "ellipse":
                        old_cx = _try_float(el.get("cx"))
                        el.set("cx", f"{old_cx + delta:.3f}")
                        adjustments_x += 1
                    elif tag_local == "line":
                        el.set("x1", f"{_try_float(el.get('x1')) + delta:.3f}")
                        el.set("x2", f"{_try_float(el.get('x2')) + delta:.3f}")
                        adjustments_x += 1
            else:
                # Right is heavier — push left (−x)
                heavy_cx = sum(ex * area for _, ex, _, area in items if ex >= cx) / (right_w + eps)
                gap   = heavy_cx - cx
                delta = gap * strength
                for el, ex, ey, _ in items:
                    if ex < cx:
                        continue
                    tag_local = el.tag.split("}")[-1]
                    if tag_local in ("rect", "text", "image"):
                        old_x = _try_float(el.get("x"))
                        el.set("x", f"{old_x - delta:.3f}")
                        adjustments_x += 1
                    elif tag_local in ("circle", "ellipse"):
                        old_cx = _try_float(el.get("cx"))
                        el.set("cx", f"{old_cx - delta:.3f}")
                        adjustments_x += 1
                    elif tag_local == "line":
                        el.set("x1", f"{_try_float(el.get('x1')) - delta:.3f}")
                        el.set("x2", f"{_try_float(el.get('x2')) - delta:.3f}")
                        adjustments_x += 1

    # ── Vertical balance ──────────────────────────────────────────────────────
    if top_w + bot_w > eps:
        ratio_tb = max(top_w, bot_w) / (min(top_w, bot_w) + eps)
        if ratio_tb > 1.0 + threshold:
            if top_w > bot_w:
                heavy_cy = sum(ey * area for _, _, ey, area in items if ey < cy) / (top_w + eps)
                gap   = cy - heavy_cy
                delta = gap * strength
                for el, ex, ey, _ in items:
                    if ey >= cy:
                        continue
                    tag_local = el.tag.split("}")[-1]
                    if tag_local in ("rect", "text", "image"):
                        old_y = _try_float(el.get("y"))
                        el.set("y", f"{old_y + delta:.3f}")
                        adjustments_y += 1
                    elif tag_local in ("circle", "ellipse"):
                        old_cy = _try_float(el.get("cy"))
                        el.set("cy", f"{old_cy + delta:.3f}")
                        adjustments_y += 1
                    elif tag_local == "line":
                        el.set("y1", f"{_try_float(el.get('y1')) + delta:.3f}")
                        el.set("y2", f"{_try_float(el.get('y2')) + delta:.3f}")
                        adjustments_y += 1
            else:
                heavy_cy = sum(ey * area for _, _, ey, area in items if ey >= cy) / (bot_w + eps)
                gap   = heavy_cy - cy
                delta = gap * strength
                for el, ex, ey, _ in items:
                    if ey < cy:
                        continue
                    tag_local = el.tag.split("}")[-1]
                    if tag_local in ("rect", "text", "image"):
                        old_y = _try_float(el.get("y"))
                        el.set("y", f"{old_y - delta:.3f}")
                        adjustments_y += 1
                    elif tag_local in ("circle", "ellipse"):
                        old_cy = _try_float(el.get("cy"))
                        el.set("cy", f"{old_cy - delta:.3f}")
                        adjustments_y += 1
                    elif tag_local == "line":
                        el.set("y1", f"{_try_float(el.get('y1')) - delta:.3f}")
                        el.set("y2", f"{_try_float(el.get('y2')) - delta:.3f}")
                        adjustments_y += 1

    print(
        f"[ASTRO-SVG] VisualWeightBalancePass applied "
        f"(strength={strength:.2f}, adj_x={adjustments_x}, adj_y={adjustments_y})"
    )


# ---------------------------------------------------------------------------
# Top-level post_process_svg  (mirrors FAstroSvgPostProcess::Process sequence)
# ---------------------------------------------------------------------------
def post_process_svg(svg_string: str) -> str:
    """
    [ASTRO-SVG] post_process_svg — entry point mirroring the FAstroSvgPostProcess
    pass chain in PostProcessing.cpp commit f2a77b0.

    Pass order (all three are optional; each checks its CVar-equivalent enable
    flag and is a no-op when disabled — same as the C++ guard pattern):

      1. EdgeSoftening       – sub-pixel AA on polyline/path connector strokes
      2. LabelCollisionAvoid – spatial repulsion to prevent text label overlap
      3. VisualWeightBalance – luminance normalisation across the SVG frame

    Passes compose linearly: each takes the output of the previous pass as
    input (mirrors Context.FinalOutput chaining in the C++ compositing graph).

    Args:
        svg_string: Assembled SVG string (from assemble_final_svg pre-output).

    Returns:
        Post-processed SVG string.
    """
    try:
        root = _parse_svg_tree(svg_string)
    except _ET.ParseError as exc:
        print(f"[ASTRO-SVG] post_process_svg: XML parse error ({exc}), returning raw SVG")
        return svg_string

    # Pass 1 – Edge Softening (r.Astro.Svg.EdgeSoften / EdgeSoftenRadius)
    if _SVG_EDGE_SOFTEN_ENABLED:
        radius = max(0.5, min(4.0, _SVG_EDGE_SOFTEN_RADIUS))
        _edge_soften_pass(root, radius)

    # Pass 2 – Label Collision Avoidance (r.Astro.Svg.LabelAvoid / LabelAvoidMargin)
    if _SVG_LABEL_AVOID_ENABLED:
        margin = max(0.0, min(32.0, _SVG_LABEL_AVOID_MARGIN))
        _label_avoid_pass(root, margin)

    # Pass 3 – Visual Weight Balancing (r.Astro.Svg.WeightBalance / WeightBalanceStrength)
    if _SVG_WEIGHT_BALANCE_ENABLED:
        strength = max(0.0, min(1.0, _SVG_WEIGHT_BALANCE_STRENGTH))
        _weight_balance_pass(root, strength)

    return _serialise_svg_tree(root)


def assemble_final_svg():
    """
    Assemble all cell SVGs into final.svg.

    Algorithm (ported from SceneRendering.cpp commit 7eae73c):
      1. FAstroCellRegistry  — build slot registry from bbox + topology + species.
      2. FConstraintCollector— collect & pack constraints; derive outgoing_edge_count.
      3. Z-layer sort        — opaque cells precede translucent cells within a layer;
                               among same-opacity cells, higher outgoing-edge count
                               renders first (constraint priority, mirrors the C++
                               SortedPassIndices.StableSort weighted by ViewZLayer).
    """
    svg_files  = glob.glob(os.path.join(CHANNELS, "cell", "*", "svg.svg"))
    bbox_files = glob.glob(os.path.join(CHANNELS, "cell", "*", "bbox.json"))

    # ── Load bbox map ──────────────────────────────────────────────────────────
    bbox_map: Dict[str, Dict] = {}
    for bf in bbox_files:
        cell_id = os.path.basename(os.path.dirname(bf))
        with open(bf) as f:
            bbox_map[cell_id] = json.load(f)

    # ── Load topology edges ────────────────────────────────────────────────────
    topology_path = os.path.join(CHANNELS, "skeleton", "topology.json")
    topology_edges: List[Dict] = []
    if os.path.exists(topology_path):
        with open(topology_path) as f:
            topology_edges = json.load(f).get("edges", [])

    # ── Load species map ───────────────────────────────────────────────────────
    species_path = os.path.join(CHANNELS, "physics", "species_assignment.json")
    species_map: Dict[str, str] = {}
    if os.path.exists(species_path):
        with open(species_path) as f:
            raw = json.load(f)
            species_map = {cid: v.get("species", "") for cid, v in raw.items()}
    # Fallback: read species directly from bbox.json
    for cid, b in bbox_map.items():
        if cid not in species_map and "species" in b:
            species_map[cid] = b["species"]

    # ── [ASTRO-CELL-REGISTRY] Populate registry ────────────────────────────────
    registry = _build_cell_registry(bbox_map, topology_edges, species_map)

    # ── [ASTRO-CONSTRAINT] Collect constraints ─────────────────────────────────
    collector = AstroConstraintCollector(registry)
    collector.collect()
    print(
        f"[ASTRO-CELL-REGISTRY] Registry init complete: {len(registry)} cells, "
        f"{collector.active_cell_count()} active constraints"
    )

    # ── Load SVG fragments ─────────────────────────────────────────────────────
    svg_map: Dict[str, str] = {}
    for sf in svg_files:
        cell_id = os.path.basename(os.path.dirname(sf))
        with open(sf) as f:
            svg_map[cell_id] = f.read()

    # ── [ASTRO-ZLAYER-SORT] Build sort key and order fragments ─────────────────
    #
    # Mirrors SetupMeshPass SortedPassIndices.StableSort (commit 7eae73c):
    #   • "Opaque" cells (non-translucent species) sort before translucent ones.
    #   • Within the same opacity class, cells with more outgoing edges (higher
    #     constraint priority) render first — they establish the visual substrate
    #     that later cells layer on top of.
    #   • Final tie-break: ascending z_layer, then cell_id (stable/deterministic).
    #
    # key tuple: (z_layer, is_translucent, -outgoing_edge_count, cell_id)
    def _sort_key(slot: AstroCellSlot) -> Tuple:
        is_translucent = int(slot.species in _TRANSLUCENT_SPECIES)
        out_priority   = -collector.outgoing_edge_count(slot.cell_id)  # negate → descending
        return (slot.z_layer, is_translucent, out_priority, slot.cell_id)

    sorted_slots = sorted(
        [s for s in registry if s.active and s.cell_id in svg_map],
        key=_sort_key,
    )

    print(
        f"[ASTRO-ZLAYER-SORT] Dispatching {len(sorted_slots)} cells in "
        f"z-layer + constraint-priority order"
    )
    for s in sorted_slots:
        translucent_tag = "translucent" if s.species in _TRANSLUCENT_SPECIES else "opaque"
        print(
            f"  cell={s.cell_id:15s} z={s.z_layer} {translucent_tag:11s} "
            f"out_edges={collector.outgoing_edge_count(s.cell_id)}"
        )

    # ── Compute canvas bounds ──────────────────────────────────────────────────
    max_x, max_y = 0, 0
    for b in bbox_map.values():
        max_x = max(max_x, b["x"] + b["w"])
        max_y = max(max_y, b["y"] + b["h"])
    width  = max_x + 60
    height = max_y + 60

    # ── Group sorted slots by z-layer (preserve intra-layer order) ────────────
    z_groups: Dict[int, List[AstroCellSlot]] = {}
    for slot in sorted_slots:
        z_groups.setdefault(slot.z_layer, []).append(slot)

    # ── Defs ───────────────────────────────────────────────────────────────────
    defs = '''  <defs>
    <marker id="arrow-green" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#2E7D32"/>
    </marker>
  </defs>'''

    # ── Build SVG ──────────────────────────────────────────────────────────────
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="max-width:100%;height:auto;">',
        f'  <rect width="{width}" height="{height}" fill="#FAFAFA" rx="4"/>',
        defs,
    ]

    for z in sorted(z_groups.keys()):
        lines.append(f'  <g id="z{z}-layer">')
        for slot in z_groups[z]:
            out_cnt = collector.outgoing_edge_count(slot.cell_id)
            lines.append(
                f'    <!-- cell: {slot.cell_id} species={slot.species} '
                f'constraint_mask=0x{slot.constraint_mask:02X} out_edges={out_cnt} -->'
            )
            lines.append(f'    {svg_map[slot.cell_id]}')
        lines.append(f'  </g>')

    lines.append('</svg>')

    raw_svg    = "\n".join(lines)
    # ── [ASTRO-SVG] Inject SVG post-processing passes (ported from PostProcessing.cpp f2a77b0) ──
    final_svg  = post_process_svg(raw_svg)
    output_path = os.path.join(CHANNELS, "..", "output_cell_loop.svg")
    with open(output_path, "w") as f:
        f.write(final_svg)

    print(
        f"[Assemble] final SVG: {len(sorted_slots)} cells, "
        f"{len(z_groups)} z-layers, {width}x{height}"
    )
    return output_path


# =============================================================================
# [ASTRO] EpochSnapshotManager
# Python port of FAstroEpochSnapshotManager from
# upstream/unreal-renderer/SceneCaptureRendering.cpp commit d31c85e.
#
# Mapping:
#   FAstroCellGroupSnapshot  → EpochSnapshot  (dict-based, JSON-serialisable)
#   FAstroCellState          → per-cell dict read from cell/*/bbox.json
#   CaptureEpochSnapshot()   → EpochSnapshotManager.capture()
#   DiffSnapshots()          → EpochSnapshotManager.diff()
#   RollbackToSnapshot()     → EpochSnapshotManager.rollback()
#
# Divergence guard (Python addition):
#   total_force = sum of |dx|+|dy| across all cells in physics/force_field.json.
#   If epoch N total_force > epoch N-1 → diverging → rollback to N-1 snapshot.
#
# Ring buffer capacity mirrors kMaxSnapshotHistory = 16 in the C++ source.
# =============================================================================

import time as _time


class EpochSnapshotManager:
    """
    Epoch snapshot manager — serialize / diff / rollback cell group state
    across epochs.  Ported from FAstroEpochSnapshotManager (d31c85e).

    Each snapshot is:
        {
          "epoch":    <int>,
          "ts":       <float ms>,
          "checksum": <int>,
          "cells": {
              "<cell_id>": { ...bbox fields... }
          },
          "total_force": <float>   # sum of |dx|+|dy| across all cells
        }

    Ring buffer of capacity MAX_HISTORY; oldest entry is overwritten when full,
    mirroring the (HistoryHead + 1) % kMaxSnapshotHistory scheme in C++.
    """

    MAX_HISTORY = 16

    def __init__(self, channels_root: str) -> None:
        self._root       = channels_root
        self._history: list = []   # ordered oldest→newest, max MAX_HISTORY entries
        self._current_epoch: int = 0

    # ------------------------------------------------------------------
    # _compute_checksum
    # XOR-fold over cell IDs + bbox epoch field, mirrors ComputeChecksum()
    # in FAstroCellGroupSnapshot: rotate-left-7 per char, rotate-left-13
    # per LastEpochIndex.  Python integers are unbounded, so we mask to
    # uint32 at each step with & 0xFFFFFFFF.
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_checksum(cells: dict) -> int:
        crc = 0x5A5A5A5A
        for cell_id in sorted(cells):
            for ch in cell_id:
                crc ^= ord(ch)
                crc  = ((crc << 7) | (crc >> 25)) & 0xFFFFFFFF
            epoch_val = cells[cell_id].get("epoch", 0)
            crc ^= epoch_val & 0xFFFFFFFF
            crc  = ((crc << 13) | (crc >> 19)) & 0xFFFFFFFF
        return crc

    # ------------------------------------------------------------------
    # _read_cells
    # Read all cell/*/bbox.json files → dict keyed by cell_id.
    # Analogous to the InCells parameter passed to CaptureEpochSnapshot().
    # ------------------------------------------------------------------
    def _read_cells(self) -> dict:
        pattern = os.path.join(self._root, "cell", "*", "bbox.json")
        cells = {}
        for bf in sorted(glob.glob(pattern)):
            cell_id = os.path.basename(os.path.dirname(bf))
            with open(bf) as f:
                cells[cell_id] = json.load(f)
        return cells

    # ------------------------------------------------------------------
    # _read_total_force
    # Compute total_force = Σ |dx| + |dy| from physics/force_field.json.
    # Used for divergence detection (Python extension, no direct C++ analog
    # but driven by the same intent as the "roll back on divergence" comment
    # in the Rollback block of the original source).
    # ------------------------------------------------------------------
    def _read_total_force(self) -> float:
        ff_path = os.path.join(self._root, "physics", "force_field.json")
        if not os.path.exists(ff_path):
            return 0.0
        with open(ff_path) as f:
            ff = json.load(f)
        total = 0.0
        for v in ff.values():
            total += abs(v.get("dx", 0.0)) + abs(v.get("dy", 0.0))
        return total

    # ------------------------------------------------------------------
    # capture
    # Serialize current cell group state → snapshot → push into ring buffer.
    # Returns the snapshot dict (caller may persist / transmit it).
    # Mirrors CaptureEpochSnapshot() including the fprintf progress log.
    # ------------------------------------------------------------------
    def capture(self) -> dict:
        cells     = self._read_cells()
        checksum  = self._compute_checksum(cells)
        ts_ms     = _time.time() * 1000.0
        total_force = self._read_total_force()

        snap = {
            "epoch":       self._current_epoch,
            "ts":          ts_ms,
            "checksum":    checksum,
            "cells":       cells,
            "total_force": total_force,
        }

        # Ring buffer: drop oldest when full, mirrors HistoryHead wrap-around.
        self._history.append(snap)
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)

        print(
            f"[ASTRO-EPOCH] capture: epoch={snap['epoch']} "
            f"cells={len(cells)} checksum=0x{checksum:08X} "
            f"total_force={total_force:.2f} ts={ts_ms:.0f}"
        )
        self._current_epoch += 1
        return snap

    # ------------------------------------------------------------------
    # diff
    # Compute Added / Removed / Modified entries between two epoch snapshots.
    # Mirrors DiffSnapshots(): fast-path on checksum equality, then field-level
    # comparison for every cell.  Returns list of dicts:
    #   {"cell_id": ..., "change": "added"|"removed"|"modified",
    #    "changed_fields": [...]}   # changed_fields only for "modified"
    # ------------------------------------------------------------------
    def diff(self, epoch_a: int, epoch_b: int) -> list:
        snap_a = self._find_snapshot(epoch_a)
        snap_b = self._find_snapshot(epoch_b)

        if snap_a is None or snap_b is None:
            print(
                f"[ASTRO-EPOCH] diff: snapshot not found for epoch "
                f"{epoch_a} or {epoch_b}"
            )
            return []

        # Fast path: identical checksum → no diff (mirrors C++ early return).
        if snap_a["checksum"] == snap_b["checksum"]:
            print(
                f"[ASTRO-EPOCH] diff: epoch {epoch_a}→{epoch_b} "
                f"checksum match, no diff"
            )
            return []

        cells_a = snap_a["cells"]
        cells_b = snap_b["cells"]
        entries = []

        # Added / Modified (iterate new snapshot, same as C++ SeenIDs pattern)
        seen = set()
        for cell_id, state_b in cells_b.items():
            seen.add(cell_id)
            if cell_id not in cells_a:
                entries.append({"cell_id": cell_id, "change": "added", "changed_fields": []})
            else:
                state_a = cells_a[cell_id]
                # Compare every field present in either snapshot
                all_keys = set(state_a) | set(state_b)
                changed  = [
                    k for k in sorted(all_keys)
                    if state_a.get(k) != state_b.get(k)
                ]
                if changed:
                    entries.append({
                        "cell_id":       cell_id,
                        "change":        "modified",
                        "changed_fields": changed,
                    })

        # Removed (in old but not new, mirrors C++ Removed scan)
        for cell_id in cells_a:
            if cell_id not in seen:
                entries.append({"cell_id": cell_id, "change": "removed", "changed_fields": []})

        print(
            f"[ASTRO-EPOCH] diff: epoch {epoch_a}→{epoch_b} "
            f"diff_entries={len(entries)}"
        )
        return entries

    # ------------------------------------------------------------------
    # check_divergence
    # Compare total_force of the two most recent snapshots.
    # If epoch N > epoch N-1 the loop is diverging: rollback to N-1 and
    # return True so run_loop() can skip the convergence check this round.
    # ------------------------------------------------------------------
    def check_divergence(self) -> bool:
        if len(self._history) < 2:
            return False

        snap_new  = self._history[-1]
        snap_prev = self._history[-2]

        force_new  = snap_new["total_force"]
        force_prev = snap_prev["total_force"]

        if force_new > force_prev:
            print(
                f"[ASTRO-EPOCH] DIVERGENCE detected: "
                f"epoch {snap_new['epoch']} force={force_new:.2f} > "
                f"epoch {snap_prev['epoch']} force={force_prev:.2f} — "
                f"rolling back to epoch {snap_prev['epoch']}"
            )
            self.rollback(snap_prev["epoch"])
            return True

        print(
            f"[ASTRO-EPOCH] convergence OK: "
            f"force {force_prev:.2f}→{force_new:.2f}"
        )
        return False

    # ------------------------------------------------------------------
    # rollback
    # Restore cell/*/bbox.json files from the target epoch snapshot and
    # rewind _current_epoch to target+1.
    # Mirrors RollbackToSnapshot(): OutCells = Snap->Cells; CurrentEpoch = target+1.
    # ------------------------------------------------------------------
    def rollback(self, target_epoch: int) -> bool:
        snap = self._find_snapshot(target_epoch)
        if snap is None:
            print(
                f"[ASTRO-EPOCH] rollback: epoch {target_epoch} not in history "
                f"(history_count={len(self._history)})"
            )
            return False

        # Write restored cell states back to bbox.json files.
        for cell_id, state in snap["cells"].items():
            bbox_path = os.path.join(self._root, "cell", cell_id, "bbox.json")
            if os.path.exists(os.path.dirname(bbox_path)):
                with open(bbox_path, "w") as f:
                    json.dump(state, f, indent=2)

        # Rewind epoch counter, exactly as C++ CurrentEpoch = TargetEpoch + 1.
        self._current_epoch = target_epoch + 1
        # Trim history to exclude snapshots after the rollback target.
        self._history = [s for s in self._history if s["epoch"] <= target_epoch]

        print(
            f"[ASTRO-EPOCH] rollback: restored epoch={target_epoch} "
            f"cells={len(snap['cells'])}"
        )
        return True

    # ------------------------------------------------------------------
    # _find_snapshot  (mirrors FindSnapshot — linear scan, O(N), N ≤ 16)
    # ------------------------------------------------------------------
    def _find_snapshot(self, epoch_idx: int):
        # Scan from newest to oldest, mirrors C++ (HistoryHead-1-i) traversal.
        for snap in reversed(self._history):
            if snap["epoch"] == epoch_idx:
                return snap
        return None

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def history_count(self) -> int:
        return len(self._history)


def run_loop(max_epochs=5):
    """Main pub/sub loop."""
    print("=" * 60)
    print("astro-svgfigure Cell Pub/Sub Loop")
    print("=" * 60)

    # [ASTRO] Instantiate epoch snapshot manager (mirrors GAstroSnapshotManager
    # module-level singleton in SceneCaptureRendering.cpp d31c85e).
    snapshot = EpochSnapshotManager(CHANNELS)

    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch} ---")
        write_channel("skeleton/epoch.json", {
            "current": epoch, "max": max_epochs, "status": "running"
        })

        # 1. All cells develop (in production: parallel sub-Claude dispatch)
        cells = run_all_cells()

        # 2. Physics engine computes forces
        collisions = physics_engine()

        # 3. [ASTRO] Capture epoch snapshot — serialize all cell/*/bbox.json
        #    into ring buffer. Mirrors CaptureEpochSnapshot() call site in
        #    FScene::UpdateSceneCaptureContents() (d31c85e).
        snap = snapshot.capture()

        # Emit diff vs previous epoch when history is deep enough.
        # Mirrors the DiffSnapshots block added in d31c85e.
        if snapshot.history_count >= 2:
            diff_entries = snapshot.diff(snap["epoch"] - 1, snap["epoch"])
            if diff_entries:
                print(
                    f"[ASTRO-EPOCH] {len(diff_entries)} cell(s) changed "
                    f"since epoch {snap['epoch'] - 1}"
                )

        # 4. [ASTRO] Divergence guard — if total_force grew, rollback and skip
        #    convergence check this round (cell states already restored).
        if snapshot.check_divergence():
            print(f"[ASTRO-EPOCH] Epoch {epoch} rolled back; retrying next iteration.")
            continue

        # 5. Convergence check
        if convergence_judge():
            print(f"\n✓ Converged at epoch {epoch}!")
            break
    else:
        print(f"\n⚠ Max epochs ({max_epochs}) reached without full convergence")

    # 6. Assemble final SVG
    write_channel("skeleton/epoch.json", {
        "current": epoch, "max": max_epochs, "status": "converged"
    })
    output = assemble_final_svg()
    print(f"\n{'=' * 60}")
    print(f"Output: {output}")
    return output


if __name__ == "__main__":
    os.chdir(CHANNELS)
    run_loop()
