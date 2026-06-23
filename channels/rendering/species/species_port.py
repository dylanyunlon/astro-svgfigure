import os, sys, json, math
from dataclasses import dataclass, field
from typing import Any, Optional
from channels.rendering.decoration.decoration_extra import _build_cell_decoration, _apply_cell_decoration_overlay, _SPECIES_NAME_TO_INDEX


def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# SpeciesParams — continuous, data-driven species definition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeciesParams:
    """Data-driven species descriptor loaded from params.json.

    Every visual / behavioural property is a continuous parameter rather than
    a hardcoded enumeration, allowing the species palette to grow without
    code changes.
    """
    primary_color:   str   = "#888888"
    glow_color:      str   = "#AAAAAA"
    glow_intensity:  float = 0.04
    algorithm_gene:  str   = "radial"
    sdf_shape:       str   = "circle"
    animation_speed: float = 0.5
    corner_radius:   float = 2.0
    opacity:         float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# params.json loader
# ═══════════════════════════════════════════════════════════════════════════════

_PARAMS_JSON = os.path.join(os.path.dirname(__file__), "params.json")

_DATACLASS_FIELDS = {f.name for f in SpeciesParams.__dataclass_fields__.values()}


def _load_params_json(path: str = _PARAMS_JSON) -> list[dict]:
    """Read the species params.json file and return the raw list of dicts."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"params.json must be a JSON array, got {type(data).__name__}")
    return data


def create_species_from_params(path: str = _PARAMS_JSON) -> dict[str, SpeciesParams]:
    """Build a ``{species_name: SpeciesParams}`` mapping from *params.json*.

    Only the eight dataclass fields are forwarded to the constructor; extra
    keys in the JSON (``color``, ``bg_color``, ``f0``, …) are preserved in
    the raw registry but not in the dataclass itself.
    """
    result: dict[str, SpeciesParams] = {}
    for entry in _load_params_json(path):
        name = entry.get("species")
        if not name:
            continue
        dc_kwargs = {k: v for k, v in entry.items() if k in _DATACLASS_FIELDS}
        result[name] = SpeciesParams(**dc_kwargs)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Backward-compatible module-level dicts rebuilt from params.json
# ═══════════════════════════════════════════════════════════════════════════════
#
# Other modules import SPECIES_METADATA, SPECIES_PALETTE,
# SPECIES_ALGORITHM_GENE, species_registry, _species_f0, and
# _species_to_index.  We reconstruct them at import time so downstream
# code is unaffected.

def _rebuild_legacy_dicts(path: str = _PARAMS_JSON):
    """Parse params.json once and emit the four legacy dicts."""
    raw = _load_params_json(path)

    metadata   : dict[str, dict[str, Any]] = {}
    palette    : dict[str, dict[str, str]] = {}
    algo_gene  : dict[str, str]            = {}
    registry   : dict[str, dict[str, Any]] = {}

    for entry in raw:
        name = entry.get("species")
        if not name:
            continue

        # SPECIES_METADATA
        metadata[name] = {
            "color":     entry.get("color",    entry.get("primary_color", "#888888")),
            "bg_color":  entry.get("bg_color", "#FFFFFF"),
            "f0":        float(entry.get("f0", entry.get("glow_intensity", 0.04))),
            "roughness": float(entry.get("roughness", 0.50)),
        }

        # SPECIES_PALETTE
        palette[name] = {
            "primary":   entry.get("primary_color",   "#888888"),
            "secondary": entry.get("secondary_color",  "#AAAAAA"),
            "glow":      entry.get("glow_color",       "#AAAAAA"),
            "shadow":    entry.get("shadow_color",      "#444444"),
        }

        # SPECIES_ALGORITHM_GENE
        algo_gene[name] = entry.get("algorithm_gene", "radial")

        # species_registry (unified)
        registry[name] = {
            **metadata[name],
            **palette[name],
            "algorithm_gene": algo_gene[name],
        }

    return metadata, palette, algo_gene, registry


SPECIES_METADATA, SPECIES_PALETTE, SPECIES_ALGORITHM_GENE, species_registry = \
    _rebuild_legacy_dicts()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions (unchanged public API)
# ═══════════════════════════════════════════════════════════════════════════════

def _species_f0(species: str) -> float:
    """Per-species F0 (normal-incidence reflectance), read from params.json."""
    entry = SPECIES_METADATA.get(species)
    if entry is not None:
        return entry["f0"]
    return 0.04


def _species_to_index(species_name: str) -> int:
    """Map a species name string to its canonical integer index."""
    return _SPECIES_NAME_TO_INDEX.get(species_name, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# SpeciesVisualDNA — complete per-species visual genome
# ═══════════════════════════════════════════════════════════════════════════════

_ASSIGNMENT_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..", "physics", "species_assignment.json"
)
_TRAITS_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..", "physics", "species_visual_traits.json"
)

# SDF shape → canonical integer token consumed by GLSL cil-*.frag
_SDF_SHAPE_TO_INT: dict[str, int] = {
    "circle":       0,
    "rect":         1,
    "rounded_rect": 2,
    "wave":         3,
    "ellipse":      4,
    "triangle":     5,
    "diamond":      6,
    "cross":        7,
}

# animation gene → float phase-increment fed into u_time scaling
_ANIM_GENE_TO_PHASE: dict[str, float] = {
    "radial":      1.00,
    "linear":      0.75,
    "grid":        0.50,
    "scatter":     1.25,
    "sinusoidal":  0.90,
    "stack":       0.60,
    "token_to_vector_heatmap":   0.65,
    "sinusoidal_wave_pattern":   0.95,
    "radial_qkv_heatmap":        1.05,
    "multi_head_ffn_flow":       0.80,
    "residual_merge_pattern":    0.70,
    "linear_softmax_projection": 0.72,
}


@dataclass
class VisualDNARecord:
    """Full visual genome for one cell_type / species pairing.

    Merges data from three sources:
      * channels/rendering/species/params.json        (per-cil-species defaults)
      * channels/physics/species_assignment.json      (cell_type → agent_derived)
      * channels/physics/species_visual_traits.json   (neighbor influence)

    All fields are ready to be forwarded to GLSL uniforms via
    :class:`SpeciesShaderBinder`.
    """
    # identity
    cell_type:        str = ""
    species_id:       str = ""          # e.g. "cil-eye"
    species_index:    int = 0           # integer token for GLSL

    # color genome
    primary_color:    str   = "#888888"  # hex RGB
    glow_color:       str   = "#AAAAAA"
    secondary_color:  str   = "#AAAAAA"
    shadow_color:     str   = "#444444"
    display_color:    str   = "#888888"  # from visual_traits (neighbor-aware)
    neighbor_color:   str   = "#AAAAAA"

    # SDF geometry
    sdf_shape:        str   = "circle"
    sdf_shape_int:    int   = 0          # pre-mapped integer for shader

    # animation genome
    animation_gene:   str   = "radial"
    animation_phase:  float = 1.0        # per-gene phase increment
    animation_speed:  float = 0.5

    # glow / bloom params
    glow_intensity:   float = 0.04
    bloom_strength:   float = 0.6        # derived from glow_intensity via mapping
    bloom_radius:     float = 1.0

    # material params
    corner_radius:    float = 2.0
    roughness:        float = 0.5
    opacity:          float = 1.0
    f0:               float = 0.04       # Fresnel reflectance at normal incidence

    # neighbor influence (from visual_traits)
    influence_strength: float = 0.0
    neighbors:          list  = field(default_factory=list)

    # raw agent_derived extras (pattern, academic_source, …)
    internal_pattern:   str  = ""
    academic_source:    str  = ""


def _hex_to_vec3(hex_color: str) -> tuple[float, float, float]:
    """Convert a #RRGGBB hex string to a normalised (r, g, b) float tuple."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (0.5, 0.5, 0.5)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


class SpeciesVisualDNA:
    """Load and expose the complete visual DNA for every known species.

    [ASTRO-SPECIES] Merges three JSON sources into a unified
    ``{cell_type: VisualDNARecord}`` registry available at
    ``SpeciesVisualDNA.registry``.

    Data sources
    ────────────
    * params.json              — per-cil-* visual defaults (color, sdf, anim)
    * species_assignment.json  — cell_type → {species, gene_traits, agent_derived}
    * species_visual_traits.json — cell_type → {display_color, neighbors, influence}

    The class is lazily initialised on first access; call ``load()`` explicitly
    to pre-warm during startup.
    """

    registry: dict[str, VisualDNARecord] = {}
    _loaded: bool = False

    # ── params.json cache (cil-species → SpeciesParams) ─────────────────────
    _params_cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def load(
        cls,
        params_path:     str = _PARAMS_JSON,
        assignment_path: str = _ASSIGNMENT_JSON,
        traits_path:     str = _TRAITS_JSON,
    ) -> None:
        """Parse all three JSON sources and populate :attr:`registry`.

        [ASTRO-SPECIES] Called once at module import or explicitly by the
        rendering pipeline before the first frame.
        """
        _dbg("ASTRO-SPECIES", "SpeciesVisualDNA.load() — reading JSON sources")

        # 1. Parse params.json into a {cil-name: raw_dict} lookup
        raw_params: dict[str, dict[str, Any]] = {}
        try:
            for entry in _load_params_json(params_path):
                name = entry.get("species", "")
                if name:
                    raw_params[name] = entry
            _dbg("ASTRO-SPECIES", f"  params.json loaded: {len(raw_params)} entries")
        except Exception as exc:  # noqa: BLE001
            _dbg("ASTRO-SPECIES", f"  params.json load failed: {exc}")

        cls._params_cache = raw_params

        # 2. Parse species_assignment.json
        raw_assign: dict[str, dict[str, Any]] = {}
        try:
            with open(assignment_path, "r", encoding="utf-8") as fh:
                raw_assign = json.load(fh)
            _dbg("ASTRO-SPECIES", f"  species_assignment.json loaded: {len(raw_assign)} entries")
        except Exception as exc:  # noqa: BLE001
            _dbg("ASTRO-SPECIES", f"  species_assignment.json load failed: {exc}")

        # 3. Parse species_visual_traits.json
        raw_traits: dict[str, dict[str, Any]] = {}
        try:
            with open(traits_path, "r", encoding="utf-8") as fh:
                raw_traits = json.load(fh)
            _dbg("ASTRO-SPECIES", f"  species_visual_traits.json loaded: {len(raw_traits)} entries")
        except Exception as exc:  # noqa: BLE001
            _dbg("ASTRO-SPECIES", f"  species_visual_traits.json load failed: {exc}")

        # 4. Merge into VisualDNARecord per cell_type
        registry: dict[str, VisualDNARecord] = {}

        for cell_type, assign in raw_assign.items():
            species_id: str = assign.get("species", "")
            agent      = assign.get("agent_derived", {})
            traits     = raw_traits.get(cell_type, {})
            params     = raw_params.get(species_id, {})

            # Color: agent_derived > params > fallback
            primary_color   = agent.get("primary_color",   params.get("primary_color",   "#888888"))
            glow_color      = agent.get("glow_color",      params.get("glow_color",       "#AAAAAA"))
            secondary_color = params.get("secondary_color", "#AAAAAA")
            shadow_color    = params.get("shadow_color",    "#444444")

            # SDF shape
            sdf_shape_str = agent.get("sdf_shape", params.get("sdf_shape", "circle"))
            sdf_shape_int = _SDF_SHAPE_TO_INT.get(sdf_shape_str, 0)

            # Animation
            anim_gene  = agent.get("algorithm_gene",  params.get("algorithm_gene",  "radial"))
            anim_speed = float(agent.get("animation_speed", params.get("animation_speed", 0.5)))
            anim_phase = _ANIM_GENE_TO_PHASE.get(anim_gene, 1.0)

            # Glow / bloom
            glow_intensity = float(params.get("glow_intensity", 0.04))
            # bloom_strength: glow_intensity drives bloom via a simple affine map:
            #   glow_intensity ∈ [0, 1] → bloom_strength ∈ [0.2, 1.8]
            bloom_strength = 0.2 + glow_intensity * 1.6
            bloom_radius   = float(params.get("roughness", 0.5))  # roughness ↔ bloom spread

            # Material
            corner_radius = float(params.get("corner_radius", 2.0))
            roughness     = float(params.get("roughness",     0.5))
            opacity       = float(params.get("opacity",       1.0))
            f0            = float(params.get("f0",            0.04))

            # Neighbor influence
            influence_strength = float(traits.get("influence_strength", 0.0))
            neighbors          = traits.get("neighbors", [])
            display_color      = traits.get("display_color",           primary_color)
            neighbor_color     = traits.get("neighbor_influence_color", glow_color)

            rec = VisualDNARecord(
                cell_type       = cell_type,
                species_id      = species_id,
                species_index   = _SPECIES_NAME_TO_INDEX.get(species_id, 0),
                primary_color   = primary_color,
                glow_color      = glow_color,
                secondary_color = secondary_color,
                shadow_color    = shadow_color,
                display_color   = display_color,
                neighbor_color  = neighbor_color,
                sdf_shape       = sdf_shape_str,
                sdf_shape_int   = sdf_shape_int,
                animation_gene  = anim_gene,
                animation_phase = anim_phase,
                animation_speed = anim_speed,
                glow_intensity  = glow_intensity,
                bloom_strength  = bloom_strength,
                bloom_radius    = bloom_radius,
                corner_radius   = corner_radius,
                roughness       = roughness,
                opacity         = opacity,
                f0              = f0,
                influence_strength = influence_strength,
                neighbors          = list(neighbors),
                internal_pattern   = agent.get("internal_pattern",  ""),
                academic_source    = agent.get("academic_source",   ""),
            )
            registry[cell_type] = rec
            _dbg(
                "ASTRO-SPECIES",
                f"  [{cell_type}] species={species_id} sdf={sdf_shape_str} "
                f"gene={anim_gene} glow={glow_intensity:.3f}",
            )

        cls.registry = registry
        cls._loaded  = True
        _dbg("ASTRO-SPECIES", f"SpeciesVisualDNA ready — {len(cls.registry)} cell types")

    @classmethod
    def _ensure_loaded(cls) -> None:
        """[ASTRO-SPECIES] Lazy-init guard: load on first access."""
        if not cls._loaded:
            cls.load()

    @classmethod
    def get(cls, cell_type: str) -> Optional[VisualDNARecord]:
        """Return the :class:`VisualDNARecord` for *cell_type*, or ``None``.

        [ASTRO-SPECIES] Triggers lazy load if the registry has not been
        populated yet.
        """
        cls._ensure_loaded()
        rec = cls.registry.get(cell_type)
        if rec is None:
            _dbg("ASTRO-SPECIES", f"SpeciesVisualDNA.get: unknown cell_type '{cell_type}'")
        return rec

    @classmethod
    def all_cell_types(cls) -> list[str]:
        """[ASTRO-SPECIES] Return sorted list of all known cell_type keys."""
        cls._ensure_loaded()
        return sorted(cls.registry.keys())

    @classmethod
    def by_species(cls, species_id: str) -> list[VisualDNARecord]:
        """[ASTRO-SPECIES] Return all DNA records that map to *species_id*."""
        cls._ensure_loaded()
        return [r for r in cls.registry.values() if r.species_id == species_id]


# Auto-load at import so the registry is ready before any render call.
try:
    SpeciesVisualDNA.load()
except Exception as _svdna_exc:  # noqa: BLE001
    _dbg("ASTRO-SPECIES", f"SpeciesVisualDNA auto-load failed (non-fatal): {_svdna_exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# SpeciesShaderBinder — bind visual DNA to GLSL uniforms
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UniformBag:
    """Flat dict of ``{uniform_name: value}`` ready to upload to a GLSL program.

    [ASTRO-SPECIES] All names match the ``uniform`` declarations in
    ``src/lib/shaders/cil-*.frag`` exactly (``u_fillColor``,
    ``u_bloomStrength``, etc.).
    """
    uniforms: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        """[ASTRO-SPECIES] Retrieve a uniform value by name."""
        return self.uniforms.get(name, default)

    def as_dict(self) -> dict[str, Any]:
        """[ASTRO-SPECIES] Return a plain copy of the uniform mapping."""
        return dict(self.uniforms)


class SpeciesShaderBinder:
    """Map a :class:`VisualDNARecord` to the GLSL uniform set for ``cil-*.frag``.

    [ASTRO-SPECIES] This class converts the visual genome (hex colors, float
    gene values, SDF shape tokens) into the exact scalar / vec3 uniform values
    expected by the family of fragment shaders in
    ``src/lib/shaders/cil-*.frag``.

    Standard interface
    ──────────────────
    ::

        dna = SpeciesVisualDNA.get("self_attn")
        bag = SpeciesShaderBinder.bind(dna)
        for name, val in bag.as_dict().items():
            gl.uniform(name, val)
    """

    # Uniform names that exist in every cil-*.frag
    _CORE_UNIFORMS = (
        "u_fillColor",       # vec3  — primary fill (normalised RGB)
        "u_glowColor",       # vec3  — glow / bloom tint
        "u_opacity",         # float — overall cell opacity
        "u_bloomStrength",   # float — homebloom/bloomStrength
        "u_bloomRadius",     # float — homebloom/bloomRadius
        "u_ambientIntensity",# float — L_Element_11 intensity
        "u_ambientColor",    # vec3  — L_Element_11 color
        "u_lightExposure",   # float — VolumetricLight fExposure
        "u_shadowFar",       # float — SHADOW_Element far plane
        "u_shadowBias",      # float — derived shadow bias
        "u_sdfShape",        # int   — SDF shape token (0=circle, 1=rect, …)
        "u_cornerRadius",    # float — rounded rect corner radius
        "u_animSpeed",       # float — animation time multiplier
        "u_animPhase",       # float — per-gene phase increment
        "u_roughness",       # float — PBR roughness
        "u_f0",              # float — Fresnel F0
        "u_speciesIndex",    # int   — species integer token
        "u_influenceStrength",# float — neighbor influence weight
    )

    @staticmethod
    def bind(
        dna: "VisualDNARecord",
        time: float = 0.0,
        extra: Optional[dict[str, Any]] = None,
    ) -> UniformBag:
        """Build a :class:`UniformBag` from *dna*.

        [ASTRO-SPECIES] Converts hex colors to normalised vec3, maps SDF shape
        strings to integer tokens, and derives secondary lighting uniforms from
        the glow genome.

        Parameters
        ──────────
        dna   : VisualDNARecord from SpeciesVisualDNA.get(cell_type)
        time  : current animation time in seconds (forwarded as u_time)
        extra : optional dict of additional uniforms to merge (override wins)

        Returns
        ───────
        UniformBag  with all standard uniforms populated.
        """
        _dbg(
            "ASTRO-SPECIES",
            f"SpeciesShaderBinder.bind: cell_type={dna.cell_type} "
            f"species={dna.species_id} t={time:.3f}",
        )

        fill_rgb    = _hex_to_vec3(dna.primary_color)
        glow_rgb    = _hex_to_vec3(dna.glow_color)
        ambient_rgb = _hex_to_vec3(dna.display_color)

        # Ambient intensity derived from bloom_strength (both drive scene energy)
        ambient_intensity = 1.0 + dna.bloom_strength * 2.0  # [1.2, 4.68] range
        # Light exposure: inverse of roughness — rougher = more diffuse, less exposure
        light_exposure = max(0.4, 1.0 - dna.roughness * 0.6)
        # Shadow: far plane scales with corner_radius as a proxy for cell size
        shadow_far  = 20.0 + dna.corner_radius * 2.5
        shadow_bias = 1.0 / (512.0 + dna.corner_radius * 100.0)

        uniforms: dict[str, Any] = {
            # Color
            "u_fillColor":        fill_rgb,
            "u_glowColor":        glow_rgb,
            "u_opacity":          dna.opacity,

            # Bloom / glow
            "u_bloomStrength":    dna.bloom_strength,
            "u_bloomRadius":      dna.bloom_radius,

            # Lighting
            "u_ambientIntensity": ambient_intensity,
            "u_ambientColor":     ambient_rgb,
            "u_lightExposure":    light_exposure,

            # Shadow
            "u_shadowFar":        shadow_far,
            "u_shadowBias":       shadow_bias,

            # SDF / geometry
            "u_sdfShape":         dna.sdf_shape_int,
            "u_cornerRadius":     dna.corner_radius,

            # Animation
            "u_time":             time,
            "u_animSpeed":        dna.animation_speed,
            "u_animPhase":        dna.animation_phase,

            # Material
            "u_roughness":        dna.roughness,
            "u_f0":               dna.f0,

            # Species metadata
            "u_speciesIndex":     dna.species_index,
            "u_influenceStrength": dna.influence_strength,
        }

        if extra:
            _dbg("ASTRO-SPECIES", f"  merging {len(extra)} extra uniforms")
            uniforms.update(extra)

        return UniformBag(uniforms=uniforms)

    @staticmethod
    def bind_for_cell_type(
        cell_type: str,
        time:  float = 0.0,
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[UniformBag]:
        """Convenience one-liner: look up DNA and bind in a single call.

        [ASTRO-SPECIES] Returns ``None`` if *cell_type* is not in the registry.
        """
        dna = SpeciesVisualDNA.get(cell_type)
        if dna is None:
            return None
        return SpeciesShaderBinder.bind(dna, time=time, extra=extra)

    @staticmethod
    def diff(bag_a: UniformBag, bag_b: UniformBag) -> dict[str, tuple[Any, Any]]:
        """[ASTRO-SPECIES] Return ``{key: (a_val, b_val)}`` for differing uniforms.

        Useful for debugging and incremental GPU upload (only push changed keys).
        """
        all_keys = set(bag_a.uniforms) | set(bag_b.uniforms)
        diffs: dict[str, tuple[Any, Any]] = {}
        for k in all_keys:
            va = bag_a.uniforms.get(k)
            vb = bag_b.uniforms.get(k)
            if va != vb:
                diffs[k] = (va, vb)
        return diffs


# ═══════════════════════════════════════════════════════════════════════════════
# SpeciesMutator — per-cell micro-variation on the base DNA
# ═══════════════════════════════════════════════════════════════════════════════

# Default magnitude of each mutation axis.  Keep these small so cells of the
# same species remain recognisably related while still being individually
# distinguishable.
_DEFAULT_MUTATION_STRENGTH: dict[str, float] = {
    "color_hue_shift":    0.04,   # ± fraction of [0, 1] hue circle
    "color_saturation":   0.06,   # ± multiplicative on S channel
    "color_value":        0.05,   # ± multiplicative on V channel
    "glow_intensity":     0.08,   # ± additive on glow_intensity
    "bloom_strength":     0.10,   # ± additive on bloom_strength
    "animation_speed":    0.07,   # ± additive on animation_speed
    "opacity":            0.04,   # ± additive on opacity
    "corner_radius":      0.30,   # ± additive in pixels
    "sdf_distort":        0.03,   # ± additive (maps to u_sdfDistort if present)
}


def _lcg_float(seed: int, seq: int) -> float:
    """Deterministic pseudo-random float in [0, 1) from an integer seed + sequence.

    [ASTRO-SPECIES] Uses a 32-bit LCG so the same cell_id always produces the
    same mutation — stable across frames without needing random state.
    """
    # LCG parameters from Numerical Recipes
    v = (seed * 1664525 + seq * 1013904223 + 0xDEADBEEF) & 0xFFFFFFFF
    v = (v ^ (v >> 16)) & 0xFFFFFFFF
    return (v & 0x00FFFFFF) / float(0x01000000)


def _lcg_signed(seed: int, seq: int) -> float:
    """[ASTRO-SPECIES] Deterministic float in [-1, 1) from seed + sequence."""
    return _lcg_float(seed, seq) * 2.0 - 1.0


def _hex_to_hsv(hex_color: str) -> tuple[float, float, float]:
    """Convert a #RRGGBB hex string to HSV in [0,1]³.

    [ASTRO-SPECIES] Used internally by SpeciesMutator to apply hue/saturation
    shifts in a perceptually uniform space before converting back to RGB.
    """
    r, g, b = _hex_to_vec3(hex_color)
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    # Value
    v = cmax

    # Saturation
    s = 0.0 if cmax == 0.0 else (delta / cmax)

    # Hue
    if delta == 0.0:
        h = 0.0
    elif cmax == r:
        h = ((g - b) / delta) % 6.0
    elif cmax == g:
        h = ((b - r) / delta) + 2.0
    else:
        h = ((r - g) / delta) + 4.0
    h = h / 6.0

    return (h % 1.0, s, v)


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    """Convert HSV in [0,1]³ back to a #RRGGBB hex string.

    [ASTRO-SPECIES] Companion to :func:`_hex_to_hsv`.
    """
    h = h % 1.0
    s = max(0.0, min(1.0, s))
    v = max(0.0, min(1.0, v))

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    if i == 0:   r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q

    ri = int(round(r * 255))
    gi = int(round(g * 255))
    bi = int(round(b * 255))
    return f"#{ri:02X}{gi:02X}{bi:02X}"


def _cell_id_to_seed(cell_id: str) -> int:
    """[ASTRO-SPECIES] Deterministic integer seed from an arbitrary cell_id string."""
    h = 0x811C9DC5  # FNV-1a 32-bit offset basis
    for ch in cell_id.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


@dataclass
class MutatedDNA:
    """A :class:`VisualDNARecord` with per-cell micro-variation applied.

    [ASTRO-SPECIES] Produced by :meth:`SpeciesMutator.mutate`.  All fields are
    independent copies of the source DNA; the original record is never mutated
    in place.
    """
    # Core identity (unchanged from base)
    cell_type:        str
    species_id:       str
    species_index:    int
    cell_id:          str   # unique per-cell identifier used as mutation seed

    # Mutated colors
    primary_color:    str
    glow_color:       str
    secondary_color:  str
    shadow_color:     str
    display_color:    str
    neighbor_color:   str

    # Mutated SDF params
    sdf_shape:        str
    sdf_shape_int:    int
    sdf_distort:      float  # additive micro-distortion for u_sdfDistort

    # Mutated animation
    animation_gene:   str
    animation_phase:  float
    animation_speed:  float

    # Mutated glow / bloom
    glow_intensity:   float
    bloom_strength:   float
    bloom_radius:     float

    # Mutated material
    corner_radius:    float
    roughness:        float
    opacity:          float
    f0:               float

    # Neighbor (unchanged)
    influence_strength: float
    neighbors:          list

    def to_uniform_bag(self, time: float = 0.0) -> UniformBag:
        """[ASTRO-SPECIES] Convenience: bind this mutated DNA to a UniformBag.

        Wraps :meth:`SpeciesShaderBinder.bind` so callers don't need to import
        the binder separately.
        """
        # Re-use SpeciesShaderBinder by duck-typing — it only accesses attrs
        return SpeciesShaderBinder.bind(self, time=time)  # type: ignore[arg-type]


class SpeciesMutator:
    """Apply deterministic per-cell micro-variation to a base :class:`VisualDNARecord`.

    [ASTRO-SPECIES] Every cell that shares a species still has a unique visual
    fingerprint: a small, repeatable hue shift, glow variance, speed jitter,
    and SDF distort.  The mutation is driven by a deterministic LCG seeded with
    the cell's identifier, so the same cell always renders identically across
    frames.

    Usage
    ─────
    ::

        base_dna = SpeciesVisualDNA.get("self_attn")
        mutated  = SpeciesMutator.mutate(base_dna, cell_id="cell_042")
        bag      = mutated.to_uniform_bag(time=t)
    """

    # Mutation strengths; callers may override per-axis via the ``strengths``
    # parameter of :meth:`mutate`.
    DEFAULT_STRENGTH = _DEFAULT_MUTATION_STRENGTH

    @staticmethod
    def mutate(
        dna:       "VisualDNARecord",
        cell_id:   str,
        strengths: Optional[dict[str, float]] = None,
    ) -> MutatedDNA:
        """Return a :class:`MutatedDNA` with small deterministic variance applied.

        [ASTRO-SPECIES] The mutation seed is derived exclusively from *cell_id*,
        so the output is stable across calls for the same cell.

        Parameters
        ──────────
        dna       : base VisualDNARecord from SpeciesVisualDNA.get()
        cell_id   : unique per-cell string (e.g. "cell_042", "body_7")
        strengths : optional override dict for per-axis mutation magnitude;
                    missing keys fall back to DEFAULT_STRENGTH

        Returns
        ───────
        MutatedDNA — a fully independent copy of *dna* with micro-variation.
        """
        seed = _cell_id_to_seed(cell_id)
        s    = {**_DEFAULT_MUTATION_STRENGTH, **(strengths or {})}

        _dbg(
            "ASTRO-SPECIES",
            f"SpeciesMutator.mutate: cell_id={cell_id!r} seed={seed:#010x} "
            f"species={dna.species_id}",
        )

        # ── Color mutation (HSV space) ─────────────────────────────────────
        def _mutate_color(hex_color: str, seq_base: int) -> str:
            h, sv, v = _hex_to_hsv(hex_color)
            h  = (h  + _lcg_signed(seed, seq_base + 0) * s["color_hue_shift"])  % 1.0
            sv = sv * (1.0 + _lcg_signed(seed, seq_base + 1) * s["color_saturation"])
            v  = v  * (1.0 + _lcg_signed(seed, seq_base + 2) * s["color_value"])
            return _hsv_to_hex(h, sv, v)

        primary_color   = _mutate_color(dna.primary_color,   seq_base=10)
        glow_color      = _mutate_color(dna.glow_color,      seq_base=20)
        secondary_color = _mutate_color(dna.secondary_color, seq_base=30)
        shadow_color    = _mutate_color(dna.shadow_color,    seq_base=40)
        display_color   = _mutate_color(dna.display_color,   seq_base=50)
        neighbor_color  = _mutate_color(dna.neighbor_color,  seq_base=60)

        # ── Glow / bloom mutation ──────────────────────────────────────────
        glow_intensity = max(0.0, dna.glow_intensity +
                             _lcg_signed(seed, 70) * s["glow_intensity"])
        bloom_strength = max(0.0, dna.bloom_strength +
                             _lcg_signed(seed, 71) * s["bloom_strength"])
        bloom_radius   = max(0.1, dna.bloom_radius +
                             _lcg_signed(seed, 72) * 0.1)

        # ── Animation mutation ─────────────────────────────────────────────
        animation_speed = max(0.05, dna.animation_speed +
                              _lcg_signed(seed, 80) * s["animation_speed"])
        # Phase is gene-specific; keep it stable — only jitter the magnitude
        animation_phase = dna.animation_phase * (
            1.0 + _lcg_signed(seed, 81) * 0.05
        )

        # ── Material mutation ──────────────────────────────────────────────
        opacity       = max(0.1, min(1.0, dna.opacity +
                                     _lcg_signed(seed, 90) * s["opacity"]))
        corner_radius = max(0.5, dna.corner_radius +
                            _lcg_signed(seed, 91) * s["corner_radius"])
        roughness     = max(0.0, min(1.0, dna.roughness +
                                     _lcg_signed(seed, 92) * 0.05))
        f0            = max(0.01, min(1.0, dna.f0 +
                                      _lcg_signed(seed, 93) * 0.01))

        # ── SDF micro-distortion ───────────────────────────────────────────
        sdf_distort = _lcg_signed(seed, 100) * s["sdf_distort"]

        _dbg(
            "ASTRO-SPECIES",
            f"  mutation deltas — glow={glow_intensity - dna.glow_intensity:+.4f} "
            f"speed={animation_speed - dna.animation_speed:+.4f} "
            f"opacity={opacity - dna.opacity:+.4f} "
            f"sdf_distort={sdf_distort:+.4f}",
        )

        return MutatedDNA(
            cell_type         = dna.cell_type,
            species_id        = dna.species_id,
            species_index     = dna.species_index,
            cell_id           = cell_id,
            primary_color     = primary_color,
            glow_color        = glow_color,
            secondary_color   = secondary_color,
            shadow_color      = shadow_color,
            display_color     = display_color,
            neighbor_color    = neighbor_color,
            sdf_shape         = dna.sdf_shape,
            sdf_shape_int     = dna.sdf_shape_int,
            sdf_distort       = sdf_distort,
            animation_gene    = dna.animation_gene,
            animation_phase   = animation_phase,
            animation_speed   = animation_speed,
            glow_intensity    = glow_intensity,
            bloom_strength    = bloom_strength,
            bloom_radius      = bloom_radius,
            corner_radius     = corner_radius,
            roughness         = roughness,
            opacity           = opacity,
            f0                = f0,
            influence_strength = dna.influence_strength,
            neighbors          = list(dna.neighbors),
        )

    @staticmethod
    def mutate_batch(
        cell_type: str,
        cell_ids:  list[str],
        strengths: Optional[dict[str, float]] = None,
    ) -> list[MutatedDNA]:
        """[ASTRO-SPECIES] Mutate one DNA record for each id in *cell_ids*.

        Convenience wrapper around :meth:`mutate` for bulk cell instantiation.
        Returns a list in the same order as *cell_ids*; unknown cell_type
        returns an empty list.
        """
        base_dna = SpeciesVisualDNA.get(cell_type)
        if base_dna is None:
            _dbg("ASTRO-SPECIES", f"mutate_batch: unknown cell_type '{cell_type}'")
            return []
        return [
            SpeciesMutator.mutate(base_dna, cell_id=cid, strengths=strengths)
            for cid in cell_ids
        ]

    @staticmethod
    def crossfade(
        dna_a: "VisualDNARecord",
        dna_b: "VisualDNARecord",
        t:     float,
        cell_id: str = "crossfade",
    ) -> MutatedDNA:
        """[ASTRO-SPECIES] Linearly interpolate between two DNA records.

        Useful for smooth transitions when a cell changes species at runtime
        (e.g. during a pubsub state change).  *t* ∈ [0, 1]: 0 = full dna_a,
        1 = full dna_b.
        """
        t = max(0.0, min(1.0, t))

        def lerp_f(a: float, b: float) -> float:
            return a + (b - a) * t

        def lerp_color(hex_a: str, hex_b: str) -> str:
            ha, sa, va = _hex_to_hsv(hex_a)
            hb, sb, vb = _hex_to_hsv(hex_b)
            # Hue interpolation on the shorter arc
            dh = hb - ha
            if dh >  0.5: dh -= 1.0
            if dh < -0.5: dh += 1.0
            return _hsv_to_hex((ha + dh * t) % 1.0, lerp_f(sa, sb), lerp_f(va, vb))

        _dbg(
            "ASTRO-SPECIES",
            f"SpeciesMutator.crossfade: {dna_a.species_id} → {dna_b.species_id} t={t:.3f}",
        )

        return MutatedDNA(
            cell_type         = dna_a.cell_type if t < 0.5 else dna_b.cell_type,
            species_id        = dna_a.species_id if t < 0.5 else dna_b.species_id,
            species_index     = dna_a.species_index if t < 0.5 else dna_b.species_index,
            cell_id           = cell_id,
            primary_color     = lerp_color(dna_a.primary_color,   dna_b.primary_color),
            glow_color        = lerp_color(dna_a.glow_color,      dna_b.glow_color),
            secondary_color   = lerp_color(dna_a.secondary_color, dna_b.secondary_color),
            shadow_color      = lerp_color(dna_a.shadow_color,    dna_b.shadow_color),
            display_color     = lerp_color(dna_a.display_color,   dna_b.display_color),
            neighbor_color    = lerp_color(dna_a.neighbor_color,  dna_b.neighbor_color),
            sdf_shape         = dna_a.sdf_shape if t < 0.5 else dna_b.sdf_shape,
            sdf_shape_int     = dna_a.sdf_shape_int if t < 0.5 else dna_b.sdf_shape_int,
            sdf_distort       = 0.0,
            animation_gene    = dna_a.animation_gene if t < 0.5 else dna_b.animation_gene,
            animation_phase   = lerp_f(dna_a.animation_phase, dna_b.animation_phase),
            animation_speed   = lerp_f(dna_a.animation_speed, dna_b.animation_speed),
            glow_intensity    = lerp_f(dna_a.glow_intensity,  dna_b.glow_intensity),
            bloom_strength    = lerp_f(dna_a.bloom_strength,  dna_b.bloom_strength),
            bloom_radius      = lerp_f(dna_a.bloom_radius,    dna_b.bloom_radius),
            corner_radius     = lerp_f(dna_a.corner_radius,   dna_b.corner_radius),
            roughness         = lerp_f(dna_a.roughness,       dna_b.roughness),
            opacity           = lerp_f(dna_a.opacity,         dna_b.opacity),
            f0                = lerp_f(dna_a.f0,              dna_b.f0),
            influence_strength = lerp_f(dna_a.influence_strength, dna_b.influence_strength),
            neighbors          = list(set(dna_a.neighbors) | set(dna_b.neighbors)),
        )
