"""
AstroCellStyleProbe — extracted from misc_extra.py to avoid loading 5000+ lines
of UE5 rendering code (path tracing, denoiser, etc.) that proc() doesn't need.
"""
import os, json
from channels.rendering.constants import _STYLE_PROBE_WEIGHT
from channels.rendering.species.species_port import _species_to_index
from channels.rendering.decoration.decoration_extra import _SPECIES_INDEX_TO_COLOUR
from channels.rendering.color.color_extra import _lerp_colour


class AstroCellStyleProbe:
    MAX_PALETTE_ENTRIES: int = 8

    def __init__(self, cell_id: str, bbox: dict, cell_style_weight: float = _STYLE_PROBE_WEIGHT):
        self.cell_id          = cell_id
        self.world_x          = float(bbox["x"])
        self.world_y          = float(bbox["y"])
        self.world_z          = float(bbox.get("z", 0))
        self.cell_w           = float(bbox["w"])
        self.cell_h           = float(bbox["h"])
        self.influence_radius = max(self.cell_w, self.cell_h) / 2.0
        self.cell_style_weight = max(0.0, min(1.0, cell_style_weight))
        self.palette: list = []
        self.dominant_species_index: int = 0

    def sample_surrounding_cells(self, channels_dir: str) -> None:
        self.palette.clear()
        self.dominant_species_index = 0
        cell_base = os.path.join(channels_dir, "cell")
        if not os.path.isdir(cell_base):
            return

        step_x = max(self.cell_w, 1.0)
        step_y = max(self.cell_h, 1.0)
        step_z = 1.0
        cx = self.world_x + self.cell_w / 2.0
        cy = self.world_y + self.cell_h / 2.0
        cz = self.world_z

        cardinal_offsets = [
            ( step_x, 0, 0), (-step_x, 0, 0),
            (0,  step_y, 0), (0, -step_y, 0),
            (0, 0,  step_z), (0, 0, -step_z),
        ]
        cardinal_tol = 0.5
        species_votes: dict = {}
        max_votes: int = 0

        for sibling in os.listdir(cell_base):
            if sibling == self.cell_id:
                continue
            bbox_path   = os.path.join(cell_base, sibling, "bbox.json")
            status_path = os.path.join(cell_base, sibling, "status.json")
            if not os.path.isfile(bbox_path):
                continue
            try:
                with open(bbox_path) as _f:
                    nbr_bbox = json.load(_f)
            except (json.JSONDecodeError, OSError):
                continue

            nbr_cx = nbr_bbox["x"] + nbr_bbox["w"] / 2.0
            nbr_cy = nbr_bbox["y"] + nbr_bbox["h"] / 2.0
            nbr_cz = float(nbr_bbox.get("z", 0))
            dx, dy, dz = nbr_cx - cx, nbr_cy - cy, nbr_cz - cz

            is_cardinal = False
            for (ox, oy, oz) in cardinal_offsets:
                if (abs(dx - ox) < cardinal_tol * step_x and
                    abs(dy - oy) < cardinal_tol * step_y and
                    abs(dz - oz) < cardinal_tol * max(step_z, 1.0)):
                    is_cardinal = True
                    break
            if not is_cardinal:
                continue

            nbr_species_name = nbr_bbox.get("species", "")
            if not nbr_species_name and os.path.isfile(status_path):
                try:
                    with open(status_path) as _f:
                        nbr_species_name = json.load(_f).get("species", "")
                except (json.JSONDecodeError, OSError):
                    pass

            nbr_species_idx = _species_to_index(nbr_species_name)
            nbr_colour = _SPECIES_INDEX_TO_COLOUR.get(nbr_species_idx,
                                                       _SPECIES_INDEX_TO_COLOUR[0])
            if len(self.palette) < self.MAX_PALETTE_ENTRIES:
                self.palette.append(nbr_colour)

            species_votes[nbr_species_idx] = species_votes.get(nbr_species_idx, 0) + 1
            if species_votes[nbr_species_idx] > max_votes:
                max_votes = species_votes[nbr_species_idx]
                self.dominant_species_index = nbr_species_idx

    def blend_toward_neighbour_palette(self, own_colour: tuple, roughness: float = 0.5) -> tuple:
        if not self.palette:
            return own_colour
        r_sum = sum(c[0] for c in self.palette)
        g_sum = sum(c[1] for c in self.palette)
        b_sum = sum(c[2] for c in self.palette)
        n = len(self.palette)
        palette_avg = (r_sum / n, g_sum / n, b_sum / n)
        inv_r = max(0.0, min(1.0, 1.0 - roughness))
        smooth = inv_r * inv_r * (3.0 - 2.0 * inv_r)
        t = smooth * self.cell_style_weight
        return _lerp_colour(own_colour, palette_avg, t)
