import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



def use_distance_field_ao() -> bool:
    """r.DistanceFieldAO && r.AOQuality >= 1。两个条件缺一不可，像人一样。"""
    return G_DISTANCE_FIELD_AO and G_DISTANCE_FIELD_AO_QUALITY >= 1





def use_ao_object_distance_field() -> bool:
    return G_AO_OBJECT_DISTANCE_FIELD and G_DISTANCE_FIELD_AO_QUALITY >= 2





# ---------------------------------------------------------------------------
# ObjectManagement — 场景 SDF 对象的增删改
# ---------------------------------------------------------------------------

def update_distance_field_object_buffers(
    scene_data: DFSceneData,
    objects_to_add: List[DFObjectBounds],
    objects_to_remove: List[int],
    surface_bias_expand: float = G_MESH_SDF_SURFACE_BIAS_EXPAND,
    parallel: bool = G_DF_PARALLEL_UPDATE,
) -> None:
    """
    对应 UpdateDistanceFieldObjectBuffers。
    先删再加，顺序不能错：若先加再删，可能误删新加的对象。
    世界上有些事情也讲究顺序，颠倒了就是另一个故事。
    """
    for idx in objects_to_remove:
        scene_data.remove_object(idx)

    # 可并行（G_DF_PARALLEL_UPDATE），此处简化为串行
    for obj in objects_to_add:
        # 表面偏移：膨胀一个 voxel 的 surface_bias_expand 比例
        expanded_radius = obj.radius * (1.0 + surface_bias_expand)
        expanded = DFObjectBounds(
            center=obj.center,
            radius=expanded_radius,
            object_index=obj.object_index,
        )
        scene_data.add_object(expanded)





def _sample_distance_field(
    obj: DFObjectBounds,
    ray_origin: Tuple[float, float, float],
    ray_dir: Tuple[float, float, float],
    t: float,
) -> float:
    """
    SDF 球体采样的极简近似。
    真正的实现要查 Atlas brick，这里用解析球替代。
    """
    px = ray_origin[0] + ray_dir[0] * t - obj.center[0]
    py = ray_origin[1] + ray_dir[1] * t - obj.center[1]
    pz = ray_origin[2] + ray_dir[2] * t - obj.center[2]
    return math.sqrt(px*px + py*py + pz*pz) - obj.radius





# ---------------------------------------------------------------------------
# 顶层接口：全帧 Distance Field AO pass
# ---------------------------------------------------------------------------

def render_distance_field_ao(
    scene_data: DFSceneData,
    pixel_positions: List[Tuple[float, float, float]],
    pixel_normals: List[Tuple[float, float, float]],
    pixel_depths: List[float],
    view_width: int,
    view_height: int,
    frustum_planes: List[Tuple[float, float, float, float]],
    view_origin: Tuple[float, float, float],
    ao_params: DFAOParameters,
    history: Optional[AOHistoryState] = None,
    frame_index: int = 0,
) -> Tuple[List[BentNormalAO], AOHistoryState]:
    """
    完整的 Distance Field AO 帧调度，对应 UE5 RenderDistanceFieldAO。

    流程：ObjectCulling → TileCulling → ScreenGridConeTrace → LightingPost。
    每一步都是上一步的筛选；最终到达屏幕的光，是经过了很多关卡的光。
    不经审查的光，称为噪点。
    """
    if not use_distance_field_ao():
        empty = [BentNormalAO() for _ in pixel_positions]
        return empty, AOHistoryState()

    # 1. 对象剔除
    culled = cull_objects_to_view(scene_data, frustum_planes, ao_params, view_origin)

    # 2. Tile cones
    tiles = build_tile_cones(view_width, view_height)

    # 3. Tile-object 交叉（scatter culling）
    tile_intersections = scatter_tile_culling(culled, tiles, scene_data)

    # 4. 收集存活的 SDF 对象
    surviving_indices = set(culled.object_indices)
    surviving_objects = [o for o in scene_data.objects if o.object_index in surviving_indices]

    # 5. Screen-grid cone trace（低分辨率）
    ao_size = get_buffer_size_for_ao(view_width, view_height)
    # 为简化，假设 pixel_positions/normals 已是低分辨率
    ao_low = compute_screen_grid_ao(
        pixel_positions=pixel_positions,
        pixel_normals=pixel_normals,
        objects=surviving_objects,
        ao_params=ao_params,
        frame_number=frame_index,
        use_history=(history is not None and history.valid),
    )

    # 6. 历史融合（depth rejection）
    hist_depths = ([b.occlusion for b in history.bent_normal_history]
                   if history and history.valid else [])
    ao_blended = update_history_depth_rejection(
        current=ao_low,
        history=history or AOHistoryState(),
        current_depths=pixel_depths,
        history_depths=hist_depths,
    )

    # 7. 空间稳定滤波
    ao_filtered = filter_history_stability(ao_blended, ao_size.x, ao_size.y)

    # 8. 上采样到全分辨率（简化：跳过，直接返回低分辨率结果）
    # geometry_aware_upsample(...) 可在此调用

    # 9. 更新历史
    new_history = update_ao_history(ao_filtered, ao_filtered, frame_index)

    return ao_filtered, new_history
# ============================================================
# Nanite CullRaster + Editor + RayTracing + Materials + Tessellation
# Ported from UE5 upstream — 鲁迅式注释穿插其中
# "世上本没有路，走的人多了，也便成了管线。"
# ============================================================

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Tuple, Any
import math


# ----------------------------------------------------------
# § 1  CullRaster — 剔除与光栅化
# ----------------------------------------------------------
# 鲁迅曾说，最好的剔除，是让看不见的东西永远看不见。
# 但GPU不懂文学，它只认布尔值。


# [ASTRO-SDF] ────────────────────────────────────────────────────────────────
# MSDFTextRenderer + GlyphCache + SDFOutline
# Ported from upstream/tiny-sdf (mapbox/tiny-sdf BSD-2) and
# upstream/activetheory-svg2msdf (Electron/msdfgen wrapper).
# Provides Python-side parameter management and glyph data coordination
# for WebGL MSDF text rendering in astro-svgfigure cell scenes.
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, math, hashlib, struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Constants — mirror tiny-sdf defaults and msdfgen CLI flags
# ---------------------------------------------------------------------------

_SDF_DEFAULT_FONT_SIZE: int    = 24          # px — tiny-sdf fontSize
_SDF_DEFAULT_BUFFER: int       = 3           # px — glyph padding (halo space)
_SDF_DEFAULT_RADIUS: int       = 8           # px — distance encoding range
_SDF_DEFAULT_CUTOFF: float     = 0.25        # inside/outside threshold
_SDF_DEFAULT_CANVAS_PADDING: int = 4         # extra px around glyph bbox

_MSDF_PXRANGE: int             = 4           # -pxrange arg for msdfgen
_MSDF_ATLAS_SIZE: int          = 128         # px — default atlas tile
_MSDF_CHANNELS: int            = 3           # RGB (MSDF) vs 1 (SDF)

_GLYPH_CACHE_MAX: int          = 4096        # max cached glyph entries
_GLYPH_CACHE_EVICT_BATCH: int  = 64         # LRU eviction batch size

_OUTLINE_DEFAULT_WIDTH: float  = 2.0         # px SDF outline half-width
_OUTLINE_DEFAULT_SOFTNESS: float = 0.5       # smoothstep softness range


def _dbg_sdf(msg: str) -> None:
    """[ASTRO-SDF] Conditional stderr debug output."""
    if os.environ.get("ASTRO_SDF_VERBOSE", "0") == "1":
        print(f"[ASTRO-SDF] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# § 1  GlyphMetrics — raw measurement data per glyph
# ---------------------------------------------------------------------------

@dataclass
class GlyphMetrics:
    """
    [ASTRO-SDF] Metrics for a single rendered glyph.

    Mirrors tiny-sdf draw() return value::
        data         — Uint8ClampedArray of SDF alpha values (width × height)
        width        — padded glyph width  (glyphWidth  + 2*buffer)
        height       — padded glyph height (glyphHeight + 2*buffer)
        glyphTop     — baseline-relative top of glyph bounding box
        glyphLeft    — left bearing
        glyphWidth   — unpadded glyph pixel width
        glyphHeight  — unpadded glyph pixel height
        glyphAdvance — horizontal advance in px at fontSize

    For MSDF (3-channel) the data array is interleaved R,G,B bytes.
    """
    char: str
    font_key: str                         # "<family>@<size>"
    data: bytes = b""                     # raw SDF/MSDF pixel bytes
    width: int = 0                        # padded width  (px)
    height: int = 0                       # padded height (px)
    glyph_top: float = 0.0
    glyph_left: float = 0.0
    glyph_width: int = 0                  # unpadded
    glyph_height: int = 0                 # unpadded
    glyph_advance: float = 0.0
    msdf: bool = False                    # True → 3-channel MSDF


# ---------------------------------------------------------------------------
# § 2  GlyphCache — LRU glyph texture cache
# ---------------------------------------------------------------------------
# Mirrors the WebGL texture atlas management implicit in mapbox-gl-js's
# GlyphAtlas and the msdf-atlas-gen output packer.
#
# Each entry is keyed by (char, font_key) → GlyphMetrics.
# When the cache exceeds _GLYPH_CACHE_MAX we evict the
# _GLYPH_CACHE_EVICT_BATCH least-recently-used entries.
# ---------------------------------------------------------------------------

class GlyphCache:
    """
    [ASTRO-SDF] LRU glyph metric cache for SDF/MSDF renderers.

    The cache stores GlyphMetrics objects keyed by (char, font_key) pairs.
    It does *not* manage GPU texture memory directly — that is the caller's
    responsibility.  This class provides lookup, insertion, eviction, and
    atlas-slot allocation metadata.

    Atlas packing strategy (mirrors msdf-atlas-gen shelf packing):
        Glyphs are packed into rows ("shelves") sorted by height.
        Each new glyph extends the current shelf or starts a new one when
        the shelf width is exhausted.

    Usage::

        cache = GlyphCache(atlas_w=512, atlas_h=512)
        metrics = cache.get('A', 'Roboto@24')
        if metrics is None:
            metrics = my_renderer.rasterize('A', 'Roboto@24')
            cache.put(metrics)
        uv = cache.atlas_uv('A', 'Roboto@24')
    """

    def __init__(self, atlas_w: int = 512, atlas_h: int = 512):
        self.atlas_w = atlas_w
        self.atlas_h = atlas_h

        # LRU order: head = most-recently-used
        self._order: List[Tuple[str, str]] = []        # [(char, font_key), ...]
        self._data: Dict[Tuple[str, str], GlyphMetrics] = {}

        # Atlas packing state
        self._shelf_x: int = 0
        self._shelf_y: int = 0
        self._shelf_h: int = 0
        # atlas positions: key → (atlas_x, atlas_y)
        self._atlas_pos: Dict[Tuple[str, str], Tuple[int, int]] = {}

        _dbg_sdf(f"GlyphCache init atlas=({atlas_w}×{atlas_h}) "
                 f"max={_GLYPH_CACHE_MAX}")

    # ── LRU helpers ──────────────────────────────────────────────────────────

    def _touch(self, key: Tuple[str, str]) -> None:
        """Move *key* to head of LRU list (most-recently-used)."""
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.insert(0, key)

    def _evict(self) -> None:
        """
        [ASTRO-SDF] Remove _GLYPH_CACHE_EVICT_BATCH least-recently-used glyphs.

        Mirrors GlyphAtlas::removeGlyphs in mapbox-gl-js — eviction
        happens in batch to amortise O(n) list surgery cost.
        Note: evicted glyph atlas regions are NOT reclaimed (no defrag).
        A full atlas reset is needed if fragmentation becomes an issue.
        """
        to_evict = self._order[_GLYPH_CACHE_MAX - _GLYPH_CACHE_EVICT_BATCH:]
        for k in to_evict:
            self._data.pop(k, None)
            self._atlas_pos.pop(k, None)
        self._order = self._order[:_GLYPH_CACHE_MAX - _GLYPH_CACHE_EVICT_BATCH]
        _dbg_sdf(f"GlyphCache evicted {len(to_evict)} glyphs "
                 f"remaining={len(self._data)}")

    # ── atlas packing ────────────────────────────────────────────────────────

    def _alloc_atlas(self, w: int, h: int) -> Optional[Tuple[int, int]]:
        """
        [ASTRO-SDF] Shelf-pack a (w × h) glyph into the atlas.
        Returns (atlas_x, atlas_y) or None if atlas is full.
        """
        if self._shelf_x + w > self.atlas_w:
            # Start new shelf
            self._shelf_y += self._shelf_h + 1
            self._shelf_x = 0
            self._shelf_h = 0

        if self._shelf_y + h > self.atlas_h:
            _dbg_sdf("GlyphCache atlas full — cannot alloc new glyph")
            return None   # atlas exhausted

        pos = (self._shelf_x, self._shelf_y)
        self._shelf_x += w + 1
        self._shelf_h = max(self._shelf_h, h)
        return pos

    # ── public API ───────────────────────────────────────────────────────────

    def get(self, char: str, font_key: str) -> Optional[GlyphMetrics]:
        """[ASTRO-SDF] Retrieve cached GlyphMetrics or None (cache miss)."""
        key = (char, font_key)
        m = self._data.get(key)
        if m is not None:
            self._touch(key)
        return m

    def put(self, metrics: GlyphMetrics) -> bool:
        """
        [ASTRO-SDF] Insert GlyphMetrics into cache with atlas slot allocation.

        Returns True on success, False if atlas is full.
        """
        key = (metrics.char, metrics.font_key)
        if key in self._data:
            self._touch(key)
            return True

        pos = self._alloc_atlas(metrics.width, metrics.height)
        if pos is None:
            return False

        self._atlas_pos[key] = pos
        self._data[key] = metrics
        self._touch(key)

        if len(self._data) > _GLYPH_CACHE_MAX:
            self._evict()

        _dbg_sdf(f"GlyphCache put char={repr(metrics.char)} "
                 f"font={metrics.font_key} atlas={pos} "
                 f"size=({metrics.width}×{metrics.height})")
        return True

    def atlas_uv(
        self,
        char: str,
        font_key: str,
    ) -> Optional[Dict[str, float]]:
        """
        [ASTRO-SDF] Return normalised UV rect {u0,v0,u1,v1} for WebGL sampling.

        Mirrors the UV computation in mapbox-gl-js GlyphAtlas.getGlyphUVData.
        """
        key = (char, font_key)
        pos = self._atlas_pos.get(key)
        m   = self._data.get(key)
        if pos is None or m is None:
            return None

        ax, ay = pos
        u0 = ax / self.atlas_w
        v0 = ay / self.atlas_h
        u1 = (ax + m.width) / self.atlas_w
        v1 = (ay + m.height) / self.atlas_h
        return {"u0": u0, "v0": v0, "u1": u1, "v1": v1}

    def stats(self) -> dict:
        """[ASTRO-SDF] Cache utilisation statistics."""
        return {
            "cached_glyphs":  len(self._data),
            "max_glyphs":     _GLYPH_CACHE_MAX,
            "atlas_w":        self.atlas_w,
            "atlas_h":        self.atlas_h,
            "shelf_y_used":   self._shelf_y + self._shelf_h,
            "atlas_fill_pct": round(
                (self._shelf_y + self._shelf_h) / max(self.atlas_h, 1) * 100, 1
            ),
        }

    def reset(self) -> None:
        """[ASTRO-SDF] Full cache and atlas reset (use after GPU texture rebuild)."""
        self._order.clear()
        self._data.clear()
        self._atlas_pos.clear()
        self._shelf_x = self._shelf_y = self._shelf_h = 0
        _dbg_sdf("GlyphCache reset")


# ---------------------------------------------------------------------------
# § 3  MSDFTextRenderer — parameter management for MSDF font rendering
# ---------------------------------------------------------------------------
# Mirrors the parameter space exposed by:
#   • tiny-sdf TinySDF constructor  (fontSize, fontFamily, buffer, radius, cutoff)
#   • activetheory-svg2msdf script  (-pxrange, -size, -autoframe, -keeporder)
#   • msdf-atlas-gen                (atlas size, charset, miter-limit)
#
# This class does NOT perform GPU rendering.  It manages the parameters
# and generates the GLSL uniform payload consumed by astro-svgfigure's
# WebGL text pass.
# ---------------------------------------------------------------------------

@dataclass
class MSDFFontParams:
    """
    [ASTRO-SDF] Full parameter set for one MSDF font configuration.

    Mirrors the combined parameter surface of tiny-sdf + msdfgen + atlas-gen.
    """
    font_family: str    = "sans-serif"
    font_size: int      = _SDF_DEFAULT_FONT_SIZE
    buffer: int         = _SDF_DEFAULT_BUFFER          # glyph padding px
    radius: int         = _SDF_DEFAULT_RADIUS          # EDT range px
    cutoff: float       = _SDF_DEFAULT_CUTOFF          # inside threshold
    pxrange: int        = _MSDF_PXRANGE                # msdfgen -pxrange
    atlas_size: int     = _MSDF_ATLAS_SIZE             # px per atlas tile
    keep_order: bool    = True                         # msdfgen -keeporder
    msdf_mode: bool     = True                         # False → single-ch SDF
    miter_limit: float  = 3.0                          # msdf-atlas-gen miter
    angle_threshold: float = 3.0                       # edge-color angle deg

    @property
    def font_key(self) -> str:
        """Stable cache key for this font configuration."""
        return f"{self.font_family}@{self.font_size}"

    def glsl_uniforms(self) -> Dict[str, Any]:
        """
        [ASTRO-SDF] Generate GLSL uniform dict for the astro-svgfigure text pass.

        Mirrors the uniform block consumed by the MSDF fragment shader::

            uniform float u_pxrange;   // smoothstep width in px
            uniform float u_cutoff;    // SDF inside/outside split
            uniform float u_buffer;    // glyph padding normalised
            uniform bool  u_msdf;      // true = MSDF, false = SDF

        The GLSL median function for MSDF::
            float median(float r,float g,float b){
                return max(min(r,g),min(max(r,g),b));
            }
            float d = median(texture.r, texture.g, texture.b) - u_cutoff;
            float alpha = clamp(d / fwidth(d) + 0.5, 0.0, 1.0);
        """
        return {
            "u_pxrange": float(self.pxrange),
            "u_cutoff":  self.cutoff,
            "u_buffer":  float(self.buffer) / max(self.atlas_size, 1),
            "u_msdf":    self.msdf_mode,
            "u_font_size": float(self.font_size),
        }


class MSDFTextRenderer:
    """
    [ASTRO-SDF] MSDF text renderer parameter manager.

    Manages one or more MSDFFontParams configurations and coordinates
    with a GlyphCache to track which glyphs have been rasterised and
    uploaded to the GPU atlas texture.

    Typical usage::

        renderer = MSDFTextRenderer()
        renderer.register_font("title", MSDFFontParams(
            font_family="Georgia", font_size=32, pxrange=6
        ))
        renderer.register_font("body", MSDFFontParams(
            font_family="Roboto", font_size=16
        ))

        # Layout a string → list of glyph render commands
        cmds = renderer.layout_text("hello", font_name="body",
                                    x=100.0, y=200.0, color=(1,1,1,1))

        # Retrieve GLSL uniform payload for shader binding
        uniforms = renderer.glsl_uniforms("body")

    Rendering pipeline mirrors tiny-sdf usage in mapbox-gl-js:
        1. layout_text()   → GlyphRenderCmd list (UV + position per char)
        2. cache.get()     → check GPU atlas hit / miss
        3. [on miss]       → caller invokes rasteriser, calls cache.put()
        4. glsl_uniforms() → bind UBO / set_uniform() calls before draw
    """

    def __init__(self, atlas_w: int = 512, atlas_h: int = 512):
        self._fonts: Dict[str, MSDFFontParams] = {}
        self.cache = GlyphCache(atlas_w=atlas_w, atlas_h=atlas_h)
        _dbg_sdf(f"MSDFTextRenderer init atlas=({atlas_w}×{atlas_h})")

    # ── font registry ────────────────────────────────────────────────────────

    def register_font(self, name: str, params: MSDFFontParams) -> None:
        """[ASTRO-SDF] Register a named font configuration."""
        self._fonts[name] = params
        _dbg_sdf(f"register_font '{name}' family={params.font_family} "
                 f"size={params.font_size} msdf={params.msdf_mode}")

    def get_params(self, font_name: str) -> MSDFFontParams:
        """[ASTRO-SDF] Retrieve params, raising KeyError on missing name."""
        if font_name not in self._fonts:
            raise KeyError(f"[ASTRO-SDF] Unknown font '{font_name}'; "
                           f"registered: {list(self._fonts)}")
        return self._fonts[font_name]

    def glsl_uniforms(self, font_name: str) -> Dict[str, Any]:
        """[ASTRO-SDF] Return GLSL uniform dict for the named font."""
        return self.get_params(font_name).glsl_uniforms()

    # ── glyph management ─────────────────────────────────────────────────────

    def ensure_glyph(
        self,
        char: str,
        font_name: str,
        rasterise_fn: Optional[Any] = None,
    ) -> Optional[GlyphMetrics]:
        """
        [ASTRO-SDF] Return GlyphMetrics for *char*, rasterising on cache miss.

        *rasterise_fn* signature::
            (char: str, params: MSDFFontParams) -> GlyphMetrics

        If *rasterise_fn* is None and the glyph is not cached, returns None.
        Mirrors GlyphAtlas::getGlyph in mapbox-gl-js.
        """
        params = self.get_params(font_name)
        metrics = self.cache.get(char, params.font_key)
        if metrics is not None:
            return metrics

        # Cache miss
        if rasterise_fn is None:
            _dbg_sdf(f"ensure_glyph miss char={repr(char)} "
                     f"font={font_name} — no rasterise_fn")
            return None

        metrics = rasterise_fn(char, params)
        if metrics is not None:
            self.cache.put(metrics)
        return metrics

    # ── text layout ──────────────────────────────────────────────────────────

    def layout_text(
        self,
        text: str,
        font_name: str,
        x: float,
        y: float,
        color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        line_height_mul: float = 1.2,
    ) -> List[Dict[str, Any]]:
        """
        [ASTRO-SDF] Layout *text* into per-glyph render command dicts.

        Each command dict contains::
            char         — the character
            font_name    — font configuration name
            x, y         — top-left pixel position of the glyph cell
            uv           — {u0,v0,u1,v1} atlas UV or None (cache miss)
            color        — RGBA tuple
            advance      — x advance after this glyph (px)
            msdf         — bool

        Newlines (\n) trigger a line break.
        Does NOT support RTL or complex shaping (Arabic/Devanagari).
        Mirrors the simple advance-cursor layout in mapbox-gl-js symbol layout.
        """
        params  = self.get_params(font_name)
        line_h  = params.font_size * line_height_mul
        cmds: List[Dict[str, Any]] = []
        cursor_x, cursor_y = x, y

        for char in text:
            if char == "\n":
                cursor_x = x
                cursor_y += line_h
                continue

            metrics = self.cache.get(char, params.font_key)
            uv      = (self.cache.atlas_uv(char, params.font_key)
                       if metrics else None)
            advance = (metrics.glyph_advance
                       if metrics else params.font_size * 0.6)

            cmds.append({
                "char":      char,
                "font_name": font_name,
                "x":         cursor_x,
                "y":         cursor_y,
                "uv":        uv,
                "color":     color,
                "advance":   advance,
                "msdf":      params.msdf_mode,
            })
            cursor_x += advance

        _dbg_sdf(f"layout_text '{text[:20]}...' "
                 f"font={font_name} cmds={len(cmds)}")
        return cmds

    # ── stats / diagnostics ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """[ASTRO-SDF] Combined renderer + cache statistics."""
        return {
            "fonts":       list(self._fonts),
            "cache_stats": self.cache.stats(),
        }


# ---------------------------------------------------------------------------
# § 4  SDFOutline — stroke/outline effect parameters
# ---------------------------------------------------------------------------
# Mirrors the GLSL outline technique described in the RESEARCH_125_svg2msdf.md
# and implemented in Three.js msdf text shaders:
#
#   float d = sdf_value - u_cutoff;
#   // Fill:
#   float fill_a  = smoothstep(-u_softness, u_softness, d);
#   // Outline:
#   float out_d   = sdf_value - (u_cutoff - u_outline_width);
#   float out_a   = smoothstep(-u_softness, u_softness, out_d) * (1.0 - fill_a);
#   vec4  color   = mix(u_outline_color, u_fill_color, fill_a);
#   gl_FragColor  = vec4(color.rgb, color.a * max(fill_a, out_a));
# ---------------------------------------------------------------------------

@dataclass
class SDFOutlineParams:
    """
    [ASTRO-SDF] Parameters for SDF/MSDF stroke outline rendering.

    Fields correspond 1-to-1 with GLSL uniforms in the outline fragment shader.
    """
    # Outline geometry
    outline_width: float  = _OUTLINE_DEFAULT_WIDTH    # px in SDF space
    softness: float       = _OUTLINE_DEFAULT_SOFTNESS # smoothstep edge width

    # Colors — RGBA in [0,1]
    fill_color: Tuple[float, float, float, float]    = (1.0, 1.0, 1.0, 1.0)
    outline_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # Shadow / glow extension (0 = disabled)
    shadow_width: float  = 0.0
    shadow_blur: float   = 0.0
    shadow_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5)
    shadow_offset_x: float = 2.0
    shadow_offset_y: float = 2.0

    # Feature toggles
    enable_outline: bool = True
    enable_shadow: bool  = False


class SDFOutline:
    """
    [ASTRO-SDF] SDF/MSDF outline and shadow effect manager.

    Generates the GLSL uniform payload for a two-pass text rendering:
        Pass 1 — shadow (if enabled): wider SDF threshold, offset UV sampling
        Pass 2 — fill + outline: sharp inner fill + smooth stroke ring

    Usage::

        outline = SDFOutline(SDFOutlineParams(
            outline_width=3.0,
            fill_color=(1.0, 0.9, 0.2, 1.0),
            outline_color=(0.1, 0.1, 0.1, 1.0),
            enable_shadow=True,
        ))
        uniforms = outline.glsl_uniforms(base_cutoff=0.25)
        # → bind uniforms to WebGL program before draw call

    GLSL fragment shader pseudo-code::

        // § MSDF median (3-channel) or plain SDF (1-channel)
        float sdf = u_msdf
            ? median(tex.r, tex.g, tex.b)
            : tex.r;

        // § Fill
        float fill_alpha = smoothstep(
            u_cutoff - u_softness,
            u_cutoff + u_softness,
            sdf
        );

        // § Outline
        float out_threshold = u_cutoff - u_outline_width / u_pxrange;
        float out_alpha = u_enable_outline
            ? smoothstep(out_threshold - u_softness,
                         out_threshold + u_softness, sdf)
              * (1.0 - fill_alpha)
            : 0.0;

        // § Shadow (separate draw call with offset UV)
        // Caller issues a second draw with u_shadow_offset applied in vertex.

        vec4 final = mix(u_outline_color, u_fill_color, fill_alpha);
        gl_FragColor = vec4(final.rgb,
                            final.a * max(fill_alpha, out_alpha));
    """

    def __init__(self, params: Optional[SDFOutlineParams] = None):
        self.params = params or SDFOutlineParams()

    def glsl_uniforms(self, base_cutoff: float = _SDF_DEFAULT_CUTOFF,
                      pxrange: int = _MSDF_PXRANGE) -> Dict[str, Any]:
        """
        [ASTRO-SDF] Build GLSL uniform dict for outline + shadow shader.

        *base_cutoff* should match MSDFFontParams.cutoff (default 0.25).
        *pxrange* should match MSDFFontParams.pxrange (default 4).
        """
        p = self.params

        # Normalise outline_width from px to SDF [0..1] space
        # outline_threshold = cutoff - outline_width / pxrange
        # (shifting the SDF iso-contour outward by outline_width pixels)
        out_threshold = base_cutoff - p.outline_width / max(pxrange, 1)

        uniforms: Dict[str, Any] = {
            # Fill + outline
            "u_cutoff":           base_cutoff,
            "u_softness":         p.softness,
            "u_fill_color":       list(p.fill_color),
            "u_outline_color":    list(p.outline_color),
            "u_outline_threshold":out_threshold,
            "u_enable_outline":   p.enable_outline,

            # Shadow
            "u_enable_shadow":    p.enable_shadow,
            "u_shadow_color":     list(p.shadow_color),
            "u_shadow_offset":    [p.shadow_offset_x, p.shadow_offset_y],
            "u_shadow_threshold": base_cutoff - (
                p.outline_width + p.shadow_width
            ) / max(pxrange, 1),
            "u_shadow_blur":      p.shadow_blur,
        }

        _dbg_sdf(f"SDFOutline uniforms cutoff={base_cutoff} "
                 f"out_thresh={out_threshold:.4f} "
                 f"shadow={p.enable_shadow}")
        return uniforms

    def preview(self) -> str:
        """
        [ASTRO-SDF] Human-readable outline spec (for debug logs / RESEARCH notes).

        Mirrors the -pxrange / threshold printout style in msdfgen CLI output.
        """
        p = self.params
        lines = [
            "[ASTRO-SDF] SDFOutline spec:",
            f"  outline_width  = {p.outline_width:.2f} px-SDF",
            f"  softness       = {p.softness:.3f}",
            f"  fill_color     = {p.fill_color}",
            f"  outline_color  = {p.outline_color}",
            f"  shadow         = {p.enable_shadow}",
        ]
        if p.enable_shadow:
            lines += [
                f"  shadow_width   = {p.shadow_width:.2f}",
                f"  shadow_blur    = {p.shadow_blur:.2f}",
                f"  shadow_offset  = ({p.shadow_offset_x},{p.shadow_offset_y})",
                f"  shadow_color   = {p.shadow_color}",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# § 5  SDF EDT helpers — Python port of tiny-sdf distance transform core
# ---------------------------------------------------------------------------
# Mirrors the Felzenszwalb/Huttenlocher EDT in mapbox/tiny-sdf index.js.
# Only the algorithmic kernel is ported; full Canvas 2D rasterisation
# is handled by the JS runtime (browser/Node) in the actual asset pipeline.
# These functions serve as reference implementations for testing and
# for headless SDF generation in CI / server-side workflows.
# ---------------------------------------------------------------------------

def _edt_1d(
    f: List[float],
    x0: int, n: int,
    d: List[float],
    v: List[int],
    z: List[float],
) -> None:
    """
    [ASTRO-SDF] 1-D EDT pass — Felzenszwalb & Huttenlocher (2012) §3.

    Computes the squared distance transform of input array *f[x0:x0+n]*
    in-place into *d[x0:x0+n]*.  *v* and *z* are pre-allocated working
    arrays of length n+1 each.

    Mirrors: edt1d() in tiny-sdf/index.js
    """
    k = 0
    v[0] = 0
    z[0] = -math.inf
    z[1] = math.inf

    for q in range(1, n):
        q_abs = x0 + q
        fq = f[q_abs]
        while True:
            r = v[k]
            s = ((fq + q_abs * q_abs) - (f[x0 + r] + r * r)) / (2 * (q_abs - r))
            if s <= z[k]:
                k -= 1
            else:
                break
        k += 1
        v[k] = q_abs - x0
        z[k] = s
        z[k + 1] = math.inf

    k = 0
    for q in range(n):
        q_abs = x0 + q
        while z[k + 1] < q_abs:
            k += 1
        r = v[k]
        dx = q_abs - (x0 + r)
        d[q_abs] = dx * dx + f[x0 + r]


def compute_sdf_alpha(
    glyph_pixels: List[float],
    width: int,
    height: int,
    radius: int = _SDF_DEFAULT_RADIUS,
    cutoff: float = _SDF_DEFAULT_CUTOFF,
) -> List[int]:
    """
    [ASTRO-SDF] Compute single-channel SDF from a binary glyph bitmap.

    *glyph_pixels* — flattened list of floats in [0.0, 1.0] (1.0 = inside).
    *width*, *height* — pixel dimensions.
    *radius* — EDT encoding range in pixels.
    *cutoff* — inside/outside threshold encoded at value 0.5 in output.

    Returns a list of ints in [0, 255] (SDF alpha channel).

    Algorithm mirrors tiny-sdf:
        gridOuter = squared EDT of exterior (background) pixels
        gridInner = squared EDT of interior (glyph) pixels
        signed_dist = sqrt(outer) - sqrt(inner)
        alpha = clamp(0.5 - signed_dist / radius + cutoff, 0, 1) * 255

    Complexity: O(width × height).
    """
    size = width * height
    grid_outer = [math.inf] * size
    grid_inner = [math.inf] * size

    for i in range(size):
        p = glyph_pixels[i]
        if p == 1.0:
            grid_outer[i] = 0.0
            grid_inner[i] = math.inf
        elif p == 0.0:
            grid_outer[i] = math.inf
            grid_inner[i] = 0.0
        else:
            # Anti-aliased pixel: partial coverage
            d = 0.5 - p
            grid_outer[i] = max(0.0, d) ** 2
            grid_inner[i] = max(0.0, -d) ** 2

    # Working arrays for 1-D EDT
    f = [0.0] * (max(width, height) + 1)
    v = [0] * (max(width, height) + 1)
    z = [0.0] * (max(width, height) + 2)

    # EDT in x-direction (rows)
    for y in range(height):
        row_start = y * width
        for x in range(width):
            f[x] = grid_outer[row_start + x]
        _edt_1d(f, 0, width, f, v, z)
        for x in range(width):
            grid_outer[row_start + x] = f[x]

        for x in range(width):
            f[x] = grid_inner[row_start + x]
        _edt_1d(f, 0, width, f, v, z)
        for x in range(width):
            grid_inner[row_start + x] = f[x]

    # EDT in y-direction (cols) — transpose trick
    for x in range(width):
        for y in range(height):
            f[y] = grid_outer[y * width + x]
        _edt_1d(f, 0, height, f, v, z)
        for y in range(height):
            grid_outer[y * width + x] = f[y]

        for y in range(height):
            f[y] = grid_inner[y * width + x]
        _edt_1d(f, 0, height, f, v, z)
        for y in range(height):
            grid_inner[y * width + x] = f[y]

    # Compose signed distance → alpha
    result: List[int] = [0] * size
    for i in range(size):
        outer_d = math.sqrt(max(0.0, grid_outer[i]))
        inner_d = math.sqrt(max(0.0, grid_inner[i]))
        signed  = outer_d - inner_d
        alpha   = max(0.0, min(1.0, 0.5 - signed / max(radius, 1) + cutoff))
        result[i] = int(round(alpha * 255))

    _dbg_sdf(f"compute_sdf_alpha ({width}×{height}) "
             f"radius={radius} cutoff={cutoff}")
    return result
