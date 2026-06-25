import json
import math
import os
import struct as _struct
import sys
from dataclasses import dataclass, field

CHANNELS = os.path.dirname(os.path.abspath(__file__))

from channels.rendering.species.species_port import (
    _species_to_index
)
from channels.rendering.decoration.decoration_extra import _SPECIES_INDEX_TO_COLOUR
from channels.rendering.color.color_extra import _colour_to_hex
from channels.rendering.styleprobe.style_probe_impl import AstroCellStyleProbe
from channels.rendering.shadow.shadow_port import compute_capsule_shadow_params
from channels.rendering.occlusion.occlusion_core import compute_crowding_opacity
from channels.rendering.shading.shading_core import compute_cell_energy_balance
from channels.rendering.registry.registry_core import (
    _load_cell_registry, register_cell_in_z_layer, update_cell_constraint
)
# from channels.rendering.shadow import * — lazy loaded
# from channels.rendering.nanite import * — lazy loaded
# from channels.rendering.passes import * — lazy loaded
# from channels.rendering.species import * — lazy loaded
# from channels.rendering.scene import * — lazy loaded
# from channels.rendering.effects import * — lazy loaded
# from channels.rendering.postprocess import * — lazy loaded
# from channels.rendering.reflection import * — lazy loaded
# from channels.rendering.lumen import * — lazy loaded
# from channels.rendering.distancefield import * — lazy loaded
# from channels.rendering.lighting import * — lazy loaded
# from channels.rendering.reflection import * — lazy loaded
# from channels.rendering.occlusion import * — lazy loaded
# from channels.rendering.resources import * — lazy loaded
# from channels.rendering.registry import * — lazy loaded
# from channels.rendering.shading import * — lazy loaded
# from channels.rendering.styleprobe import * — lazy loaded
# from channels.rendering.translucency import * — lazy loaded
# from channels.rendering.species import * — lazy loaded
# from channels.rendering.postprocess import * — lazy loaded
# from channels.rendering.compositor import * — lazy loaded
# from channels.rendering.effects import * — lazy loaded
# from channels.rendering.scene import * — lazy loaded
# from channels.rendering.passes import * — lazy loaded
# from channels.rendering.acceleration import * — lazy loaded
# from channels.rendering.nanite import * — lazy loaded
# from channels.rendering.lumen import * — lazy loaded
# from channels.rendering.distancefield import * — lazy loaded
# from channels.rendering.streaming import * — lazy loaded
# from channels.rendering.lighting import * — lazy loaded
# from channels.rendering.shadow import * — lazy loaded

# All rendering logic split into channels/rendering/
# rendering modules imported explicitly where needed

# ═══════════════════════════════════════════════

def read_channel(path: str) -> dict:
    """Subscribe = read JSON. Apollo CreateReader equivalent."""
    full = os.path.join(CHANNELS, path)
    with open(full) as f:
        return json.load(f)





# ═══════════════════════════════════════════════



def proc(cell_id: str):
    """
    Apollo Component::Proc() equivalent.
    Reads channels → computes visual params → publishes JSON (SVG 由主 Claude 绝对静止模式生成).

    L3: If channels/cell/{cell_id}/agent_params.json exists (written by
    dispatch_cell_agent before this call), the agent's bbox and opacity
    override the default skeleton+force-field values.  This is the final
    link in the L3 chain:
        agent computes params → agent_params.json → proc() renders with them.
    """
    # ── Subscribe: read channels ──
    skeleton = read_channel(f"skeleton/cell/{cell_id}.json")
    force_field = read_channel("physics/force_field.json")
    z_layers = read_channel("physics/z_layers.json")

    label = skeleton["label"]
    species = skeleton["species"]
    gene_traits = skeleton["gene_traits"]
    bbox = skeleton["initial_bbox"].copy()

    # ── [M371] Hierarchical compound proc ───────────────────────────────────
    # Detect compound parent (has children_ids) and child cells (has parent_id).
    #
    # Compound parent:
    #   When skeleton declares children_ids, this cell is a transparent
    #   container.  Its bbox is computed as union(children bboxes) + 20px
    #   padding on each side.  The proc short-circuits after publishing
    #   bbox/status/params — no species rendering, no shadow, no crowding.
    #
    # Child cell:
    #   When skeleton declares parent_id, the child's position is relative
    #   to the parent's published origin.  The parent's (x, y) is read from
    #   channels/cell/{parent_id}/bbox.json and added as an offset so the
    #   child's absolute position = parent_origin + child_local_position.
    # ──────────────────────────────────────────────────────────────────────────
    _COMPOUND_PADDING = 20  # px on each side of the union bbox

    children_ids = skeleton.get("children_ids", None)
    parent_id = skeleton.get("parent_id", None)

    # ── Child cell: convert local position to absolute ────────────────────
    if parent_id is not None:
        _parent_bbox_path = os.path.join(CHANNELS, "cell", parent_id, "bbox.json")
        if os.path.isfile(_parent_bbox_path):
            try:
                with open(_parent_bbox_path) as _pbf:
                    _parent_bbox = json.load(_pbf)
                # Child position is relative to parent origin — offset by
                # parent's top-left corner (+ padding to stay inside container)
                bbox["x"] += _parent_bbox["x"] + _COMPOUND_PADDING
                bbox["y"] += _parent_bbox["y"] + _COMPOUND_PADDING
                print(
                    f"[M371] Child cell_id={cell_id} parent_id={parent_id} "
                    f"offset by parent origin ({_parent_bbox['x']}, {_parent_bbox['y']}) "
                    f"→ absolute ({bbox['x']}, {bbox['y']})",
                    file=sys.stderr,
                )
            except Exception as _pe:
                print(
                    f"[M371] WARNING: failed to read parent bbox for "
                    f"cell_id={cell_id} parent_id={parent_id}: {_pe}",
                    file=sys.stderr,
                )

    # ── Compound parent: wait for children, compute union bbox ────────────
    if children_ids is not None and len(children_ids) > 0:
        # Collect children bboxes — children must have been processed first
        # (run_all_cells topological order guarantees children before parents
        # when compound groups are in the topology).
        _child_bboxes = []
        for _cid in children_ids:
            _child_bbox_path = os.path.join(CHANNELS, "cell", _cid, "bbox.json")
            if os.path.isfile(_child_bbox_path):
                try:
                    with open(_child_bbox_path) as _cbf:
                        _child_bboxes.append(json.load(_cbf))
                except (json.JSONDecodeError, OSError) as _ce:
                    print(
                        f"[M371] WARNING: cannot read child bbox "
                        f"child_id={_cid}: {_ce}",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"[M371] WARNING: child bbox not yet published "
                    f"child_id={_cid} — skipping in union",
                    file=sys.stderr,
                )

        if _child_bboxes:
            # Union of all children bboxes
            _min_x = min(cb["x"] for cb in _child_bboxes)
            _min_y = min(cb["y"] for cb in _child_bboxes)
            _max_x = max(cb["x"] + cb["w"] for cb in _child_bboxes)
            _max_y = max(cb["y"] + cb["h"] for cb in _child_bboxes)

            # Inflate by padding on each side
            bbox["x"] = _min_x - _COMPOUND_PADDING
            bbox["y"] = _min_y - _COMPOUND_PADDING
            bbox["w"] = (_max_x - _min_x) + 2 * _COMPOUND_PADDING
            bbox["h"] = (_max_y - _min_y) + 2 * _COMPOUND_PADDING
            # z: compound container sits behind its children
            bbox["z"] = min((cb.get("z", 3) for cb in _child_bboxes), default=3) - 1

        print(
            f"[M371] Compound parent cell_id={cell_id} "
            f"children={children_ids} collected={len(_child_bboxes)} "
            f"union bbox=({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) "
            f"z={bbox['z']}",
            file=sys.stderr,
        )

        # ── Short-circuit: publish as transparent container ────────────────
        # No species rendering, no shadow, no crowding opacity — just a
        # transparent grouping rectangle with dashed stroke.
        current_epoch = read_channel("skeleton/epoch.json")["current"]
        cell_dir = f"cell/{cell_id}"
        write_channel(f"{cell_dir}/bbox.json", {
            "x": bbox["x"], "y": bbox["y"],
            "w": bbox["w"], "h": bbox["h"],
            "z": bbox["z"],
            "species": species,
            "epoch": current_epoch,
            "compound": True,
            "children_ids": children_ids,
        })
        write_channel(f"{cell_dir}/status.json", {
            "status": "converged",
            "cell_id": cell_id,
            "species": species,
            "epoch": current_epoch,
            "compound": True,
        })
        _compound_params: dict = {
            "cell_id":        cell_id,
            "species":        species,
            "compound":       True,
            "children_ids":   children_ids,
            "bbox":           {
                "x": bbox["x"], "y": bbox["y"],
                "w": bbox["w"], "h": bbox["h"],
                "z": bbox["z"],
            },
            "z":              bbox["z"],
            "opacity":        0.0,
            "fill_color":     "none",
            "stroke_color":   "#888888",
            "stroke_dash":    "6,3",
            "label":          label,
            "font_size":      10,
            "species_params": {},
            "epoch":          current_epoch,
            "shadow":         {"dx": 0, "dy": 0, "blur": 0, "opacity": 0},
        }
        write_channel(f"{cell_dir}/params.json", _compound_params)

        # Register in cell_registry
        published_bbox = {
            "x": bbox["x"], "y": bbox["y"],
            "w": bbox["w"], "h": bbox["h"],
            "z": bbox["z"],
        }
        registry = _load_cell_registry()
        if cell_id in registry["cells"]:
            update_cell_constraint(cell_id, published_bbox, current_epoch)
        else:
            register_cell_in_z_layer(cell_id, published_bbox, species, current_epoch)

        print(
            f"[Cell {cell_id}] COMPOUND container "
            f"bbox=({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) "
            f"z={bbox['z']} children={children_ids}",
        )
        return None
    # ── End [M371] compound proc ──────────────────────────────────────────

    # Apply force field adjustments
    force = force_field.get(cell_id, {"dx": 0, "dy": 0, "dz": 0})
    bbox["x"] += force["dx"]
    bbox["y"] += force["dy"]
    bbox["z"] = z_layers.get(cell_id, 3) + force.get("dz", 0)

    # ── L3: apply agent_params if present ────────────────────────────────────
    # dispatch_cell_agent() writes agent_params.json before proc() is called
    # by run_all_cells().  When the file exists, the agent's decisions win over
    # the default force-field values for bbox and opacity.
    _agent_opacity: float | None = None
    _agent_params_path = os.path.join(CHANNELS, "cell", cell_id, "agent_params.json")
    if os.path.isfile(_agent_params_path):
        try:
            with open(_agent_params_path) as _apf:
                _agent_params = json.load(_apf)
            # Override bbox if agent provided one
            if "bbox" in _agent_params and isinstance(_agent_params["bbox"], dict):
                _ab = _agent_params["bbox"]
                for _k in ("x", "y", "w", "h", "z"):
                    if _k in _ab:
                        bbox[_k] = _ab[_k]
            # Record agent opacity for use below (blended with crowding_opacity)
            if "opacity" in _agent_params and isinstance(_agent_params["opacity"], (int, float)):
                _agent_opacity = max(0.35, min(1.0, float(_agent_params["opacity"])))
            # import sys  # already imported at module level
            print(
                f"[proc] L3 agent_params applied: cell_id={cell_id} "
                f"bbox=({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) "
                f"z={bbox['z']} opacity={_agent_opacity}",
                file=sys.stderr,
            )
        except Exception as _ap_exc:
            # import sys  # already imported at module level
            print(f"[proc] WARNING: failed to read agent_params.json "
                  f"for cell_id={cell_id}: {_ap_exc}", file=sys.stderr)

    # ── [ASTRO-CELL] ReflectionEnvironment — cell style probe consistency ─────
    # Port of FAstroCellStyleProbe::SampleSurroundingCells + BlendWithCubemap
    # from commit 5d07a0a (upstream/unreal-renderer/ReflectionEnvironment.cpp).
    #
    # Build a probe for this cell, sample its six cardinal neighbours from the
    # already-published cell/*/bbox.json + status.json channels, then nudge the
    # primary SVG stroke/fill colour of this cell 20 % toward the neighbourhood
    # palette average (鲁迅式: enough to feel the pull, not enough to surrender).
    #
    # roughness maps to the cell's visual complexity:
    #   simple icon species (cil-eye, cil-bolt) → low roughness → stronger pull
    #   composite / arrow species → higher roughness → weaker pull
    _SPECIES_ROUGHNESS: dict = {
        "cil-eye":         0.1,   # smooth focal icon — very susceptible
        "cil-bolt":        0.2,   # sharp energy icon — susceptible
        "cil-plus":        0.3,   # structured cross — moderate
        "cil-vector":      0.5,   # multi-arrow — moderate resistance
        "cil-arrow-right": 0.7,   # directional terminal — mostly independent
        # ── New species ──────────────────────────────────────────────────────
        "cil-filter":      0.3,   # grid wireframe — structured, moderate
        "cil-code":        0.4,   # brace icon — moderate
        "cil-layers":      0.2,   # stacked rects — smooth depth signal
        "cil-loop":        0.5,   # arc arrow — moderate self-expression
        "cil-graph":       0.6,   # node-edge graph — relatively independent
    }
    _probe_roughness = _SPECIES_ROUGHNESS.get(species, 0.5)

    _style_probe = AstroCellStyleProbe(cell_id, bbox)
    _style_probe.sample_surrounding_cells(CHANNELS)

    # Own primary colour (species fill colour from _SPECIES_INDEX_TO_COLOUR)
    _own_species_idx   = _species_to_index(species)
    _own_colour_rgb    = _SPECIES_INDEX_TO_COLOUR.get(_own_species_idx,
                                                      _SPECIES_INDEX_TO_COLOUR[0])
    _blended_colour_rgb = _style_probe.blend_toward_neighbour_palette(
        _own_colour_rgb, roughness=_probe_roughness)
    _blended_hex = _colour_to_hex(_blended_colour_rgb)

    # Log probe result — mirrors UE_LOG VeryVerbose in CubemapSlot loop.
    print(
        f"[ASTRO-CELL] StyleProbe cell_id={cell_id} "
        f"dominant_species={_style_probe.dominant_species_index} "
        f"palette_entries={len(_style_probe.palette)} "
        f"weight={_style_probe.cell_style_weight:.2f} "
        f"own={_colour_to_hex(_own_colour_rgb)} "
        f"blended={_blended_hex} "
        f"roughness={_probe_roughness:.1f}",
        file=sys.stderr,
    )

    # ── [ASTRO-CELL] Capsule shadow: collect all sibling cell bboxes ─────────
    # BuildCellOcclusionVolumes equivalent: load every published bbox to build
    # the occluder set (cells that haven't published yet are silently skipped).
    all_bboxes: dict = {}
    cell_base = os.path.join(CHANNELS, "cell")
    if os.path.isdir(cell_base):
        for sibling in os.listdir(cell_base):
            bbox_path = os.path.join(cell_base, sibling, "bbox.json")
            if os.path.isfile(bbox_path):
                try:
                    with open(bbox_path) as _f:
                        all_bboxes[sibling] = json.load(_f)
                except (json.JSONDecodeError, OSError):
                    pass
    # Also include self so occlusion volume is built consistently
    all_bboxes[cell_id] = {
        "x": bbox["x"], "y": bbox["y"],
        "w": bbox["w"], "h": bbox["h"],
        "z": bbox["z"],
    }

    # ProjectCellOcclusion aggregation → SVG filter parameters
    sp = compute_capsule_shadow_params(cell_id, bbox, all_bboxes)
    shadow_dx      = sp["dx"]
    shadow_dy      = sp["dy"]
    shadow_blur    = sp["blur"]
    shadow_opacity = sp["opacity"]

    # ── [ASTRO-CELL] PostProcessAO — crowding attenuation (commit 33e27b7) ────
    # FAstroConstraintAO::ComputeConstraintWeight() port.
    # Reads neighbour bboxes as the SSAO kernel; computes 3-pass constraint-
    # space AO weight; maps to fill opacity on the cell's outermost <g>.
    # High neighbour density → lower opacity (crowd suppresses the cell's
    # visual weight, preventing the SVG equivalent of SSAO black halos in
    # packed cell regions).
    crowding_opacity = compute_crowding_opacity(cell_id, bbox, all_bboxes)

    # ── L3: agent opacity override ────────────────────────────────────────────
    # If dispatch_cell_agent provided an opacity, blend it with the AO-derived
    # crowding_opacity: agent 60% authority, PostProcessAO 40%.  This lets the
    # agent express intent (e.g. highlight a focal cell) while still respecting
    # neighbourhood density cues.
    if _agent_opacity is not None:
        crowding_opacity = 0.6 * _agent_opacity + 0.4 * crowding_opacity

    # ── [ASTRO-CELL] ShadingEnergyConservation — colour energy balance ────────
    # Port of ShadingEnergyConservation.cpp (upstream/unreal-renderer-ue5).
    # Replaces LUT texture reads with analytic GGX/Schlick/cloth approximations.
    # Ensures fill + stroke + shadow energy budget <= 1.0 (furnace-test).
    # Fresnel edge-glow compensation brightens stroke at grazing angles.
    # Energy preservation attenuates fill by specular directional albedo.
    _SPECIES_ROUGHNESS_SEC: dict = {
        "cil-eye":         0.1,
        "cil-bolt":        0.2,
        "cil-plus":        0.3,
        "cil-vector":      0.5,
        "cil-arrow-right": 0.7,
        "cil-filter":      0.3,
        "cil-code":        0.4,
        "cil-layers":      0.2,
        "cil-loop":        0.5,
        "cil-graph":       0.6,
    }
    _sec_roughness   = _SPECIES_ROUGHNESS_SEC.get(species, 0.5)
    _stroke_opacity  = 0.85    # species stroke default (pre-conservation)
    _sec_result = compute_cell_energy_balance(
        cell_id        = cell_id,
        species        = species,
        bbox           = bbox,
        fill_opacity   = crowding_opacity,
        stroke_opacity = _stroke_opacity,
        shadow_opacity = shadow_opacity,
        roughness      = _sec_roughness,
    )
    crowding_opacity = _sec_result["fill_opacity"]
    shadow_opacity   = _sec_result["shadow_opacity"]

    # ── Publish: write to channels ──
    current_epoch = read_channel("skeleton/epoch.json")["current"]
    cell_dir = f"cell/{cell_id}"
    write_channel(f"{cell_dir}/bbox.json", {
        "x": bbox["x"], "y": bbox["y"],
        "w": bbox["w"], "h": bbox["h"],
        "z": bbox["z"],
        "species": species,
        "epoch": current_epoch,
    })
    write_channel(f"{cell_dir}/status.json", {
        "status": "converged",
        "cell_id": cell_id,
        "species": species,
        "epoch": current_epoch,
    })

    # ── [M008] params.json — PixiJS frontend parameter channel ───────────────
    # Primary output path for PixiJS renderer: all drawing parameters are
    # surfaced here so the frontend never needs to parse SVG.
    # species_params encodes the species-specific visual constants that the
    # PixiJS draw call needs (ring_count / pupil_radius for cil-eye, etc.).
    # blended fill/stroke colours are extracted from the SVG colour pipeline
    # above (post StyleProbe blend) so the frontend receives the final values.
    _species_params: dict
    if species == "cil-eye":
        _num_rays = max(4, len(label) // 2)
        _r_outer_eye = min(bbox["w"], bbox["h"]) / 2 - 4
        _species_params = {
            "ring_count":    _num_rays,
            "pupil_radius":  round(_r_outer_eye * 0.2, 2),
            "r_outer":       round(_r_outer_eye, 2),
            "r_inner_ratio": 0.3,
        }
    elif species == "cil-bolt":
        _zigzag_segments = 6
        _seg_w = (bbox["w"] - 20) / _zigzag_segments
        _species_params = {
            "zigzag_count": _zigzag_segments,
            "amplitude":    6.0,
            "seg_width":    round(_seg_w, 2),
        }
    elif species == "cil-vector":
        _num_arrows_v = 5
        _arrow_len_v  = bbox["w"] * 0.3
        _species_params = {
            "arrow_count":   _num_arrows_v,
            "arrow_length":  round(_arrow_len_v, 2),
            "angle_spread":  0.8,
        }
    elif species == "cil-plus":
        _arm_plus = min(bbox["w"], bbox["h"]) * 0.25
        _species_params = {
            "arm_length":    round(_arm_plus, 2),
            "stroke_width":  2.5,
            "dash_corners":  True,
        }
    elif species == "cil-arrow-right":
        _aw = bbox["w"] * 0.3
        _species_params = {
            "arrow_width":   round(_aw, 2),
            "arrow_height":  16.0,
        }
    elif species == "cil-filter":
        _pad_f  = max(8, min(bbox["w"], bbox["h"]) * 0.12)
        _cell_w = (bbox["w"] - 2 * _pad_f) / 3
        _cell_h = ((bbox["h"] - 2 * _pad_f) * 0.72) / 3
        _species_params = {
            "grid_cols":   3,
            "grid_rows":   3,
            "cell_width":  round(_cell_w, 2),
            "cell_height": round(_cell_h, 2),
            "pad":         round(_pad_f, 2),
        }
    elif species == "cil-code":
        _arm_c = min(bbox["w"], bbox["h"]) * 0.28
        _species_params = {
            "brace_arm":    round(_arm_c, 2),
            "nib_ratio":    0.22,
            "corner_radius_ratio": 0.35,
        }
    elif species == "cil-layers":
        _pad_l = max(6, min(bbox["w"], bbox["h"]) * 0.10)
        _rh_l  = (bbox["h"] - 2 * _pad_l) * 0.48
        _step_l = _rh_l * 0.32
        _species_params = {
            "layer_count":  3,
            "layer_height": round(_rh_l, 2),
            "stagger_step": round(_step_l, 2),
            "opacities":    [0.35, 0.50, 0.68],
        }
    elif species == "cil-loop":
        _r_loop = min(bbox["w"], bbox["h"]) * 0.28
        _species_params = {
            "arc_radius":   round(_r_loop, 2),
            "gap_degrees":  60,
            "sweep_cw":     True,
        }
    elif species == "cil-graph":
        _r_outer_g = min(bbox["w"], bbox["h"]) * 0.28
        _r_inner_g = _r_outer_g * 0.38
        _node_r    = max(3.0, min(bbox["w"], bbox["h"]) * 0.055)
        _species_params = {
            "node_count":   5,
            "r_outer":      round(_r_outer_g, 2),
            "r_inner":      round(_r_inner_g, 2),
            "node_radius":  round(_node_r, 2),
            "edge_list":    [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [3, 4]],
        }
    else:
        _species_params = {}

    # Derive final fill/stroke colours from blended hex (post-StyleProbe)
    # _blended_hex is the post-blend primary fill; stroke uses species defaults.
    _stroke_colour = _colour_to_hex(
        _SPECIES_INDEX_TO_COLOUR.get(_own_species_idx, _SPECIES_INDEX_TO_COLOUR[0])
    )

    _params_payload: dict = {
        "cell_id":        cell_id,
        "species":        species,
        "bbox":           {
            "x": bbox["x"], "y": bbox["y"],
            "w": bbox["w"], "h": bbox["h"],
            "z": bbox["z"],
        },
        "z":              bbox["z"],
        "opacity":        round(crowding_opacity, 4),
        "fill_color":     _blended_hex,
        "stroke_color":   _stroke_colour,
        "label":          label,
        "font_size":      10,
        "species_params": _species_params,
        "epoch":          current_epoch,
        # Shadow params for PixiJS drop-shadow filter
        "shadow": {
            "dx":      shadow_dx,
            "dy":      shadow_dy,
            "blur":    shadow_blur,
            "opacity": shadow_opacity,
        },
    }
    write_channel(f"{cell_dir}/params.json", _params_payload)
    print(
        f"[M008] params.json written: cell_id={cell_id} "
        f"species={species} fill={_blended_hex} "
        f"species_params_keys={list(_species_params.keys())}",
        file=sys.stderr,
    )

    # -------------------------------------------------------------------------
    # [ASTRO-CELL] AddCell / UpdateCellConstraint — register the published bbox
    # in the global cell_registry channel (GAstroCellZLayerRegistry +
    # GAstroCellProxyMap pub/sub equivalent).
    #
    # Decision mirrors the render-thread paths in RendererScene.cpp 72c4d0c:
    #
    #   • First publish (no prior entry in registry)
    #       → AddPrimitiveSceneInfo_RenderThread → AstroRegisterCellInZLayer
    #         Creates a FAstroCellSceneProxy, assigns z-layer bucket, appends
    #         to GAstroCellZLayerRegistry[layer], stores in GAstroCellProxyMap.
    #
    #   • Re-publish (entry already present — bounds may have changed)
    #       → UpdatePrimitiveTransform_RenderThread → AstroUpdateCellConstraint
    #         Checks HasChanged(Tolerance=0.01); if changed, migrates proxy
    #         across z-layer buckets if necessary and sets bDirty=true
    #         (constraint_mask=1) to signal a pending constraint-buffer flush.
    #
    # All subsequent cells can read cell_registry.json to obtain the latest
    # bbox, species, z-layer, and dirty state for every peer — the Apollo
    # "scene graph" global state view.
    # -------------------------------------------------------------------------
    published_bbox = {
        "x": bbox["x"], "y": bbox["y"],
        "w": bbox["w"], "h": bbox["h"],
        "z": bbox["z"],
    }
    registry = _load_cell_registry()
    if cell_id in registry["cells"]:
        # UpdateCellConstraint path — cell already registered, check for drift
        update_cell_constraint(cell_id, published_bbox, current_epoch)
    else:
        # AddCell path — first time this cell enters the scene graph
        register_cell_in_z_layer(cell_id, published_bbox, species, current_epoch)

    print(f"[Cell {cell_id}] species={species} bbox=({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) z={bbox['z']} "
          f"shadow(dx={shadow_dx},dy={shadow_dy},blur={shadow_blur},opacity={shadow_opacity}) "
          f"crowding_opacity={crowding_opacity}")
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 cell_component.py <cell_id>")
        print("  e.g. python3 cell_component.py self_attn")
        sys.exit(1)
    cell_id = sys.argv[1]
    proc(cell_id)


# Lumen/Lighting/Shadow re-exports — lazy loaded to avoid circular imports
# and 5000+ line UE5 rendering modules that proc() doesn't need.
# Import on demand: from channels.rendering.lumen import ...
# Import on demand: from channels.rendering.lighting import ...
# Import on demand: from channels.rendering.shadow import ...













def write_channel(path: str, data):
    """Publish = write file. Apollo CreateWriter equivalent."""
    full = os.path.join(CHANNELS, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if isinstance(data, str):
        with open(full, "w") as f:
            f.write(data)
    else:
        with open(full, "w") as f:
            json.dump(data, f, indent=2)

# ═══════════════════════════════════════════════
# Species Gene Algorithms — each species generates differently
# These are NOT icon files — they are algorithmic generation styles
# ═══════════════════════════════════════════════



# ═══════════════════════════════════════════════



