import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



def should_render_light_shafts(species: str) -> bool:
    """
    Mirrors ShouldRenderLightShafts(FSceneViewFamily&).

    Returns True when light shafts are globally enabled and the species has
    «light shaft emissive» semantics (bright focal point or energy spike).

    鲁迅式：不是所有光都值得拖成光柱——只有足够明亮的，才有资格散射。
    """
    if not ASTRO_LIGHT_SHAFTS_ENABLED:
        return False
    _LIGHT_SHAFT_SPECIES = {"cil-eye", "cil-bolt", "cil-loop"}
    return species in _LIGHT_SHAFT_SPECIES





def should_render_light_shafts_for_cell(
    cell_id: str,
    bbox:    dict,
    viewport_w: float = 1200.0,
    viewport_h: float = 900.0,
) -> bool:
    """
    Mirrors ShouldRenderLightShaftsForLight() — returns False if the light
    source (cell) is behind the viewer (screen-space W ≤ 0).

    In 2-D, «behind the viewer» means the cell centre is outside the viewport
    with a very large negative Z-offset (no such cells in practice).

    鲁迅式：只有在观众视野之内的光源，才有资格投下光柱。
    """
    cx = bbox.get("x", 0) + bbox.get("w", 100) / 2.0
    cy = bbox.get("y", 0) + bbox.get("h", 50)  / 2.0
    # Treat the viewport bounds as the «W > 0» constraint
    return (0 <= cx <= viewport_w) and (0 <= cy <= viewport_h)





def get_cell_light_shaft_params(
    cell_id:    str,
    bbox:       dict,
    viewport_w: float = 1200.0,
    viewport_h: float = 900.0,
) -> dict:
    """
    Mirrors GetLightShaftParameters() — computes the FLightShaftPixelShaderParameters
    equivalent for a single cell acting as a directional light shaft emitter.

    Returns a dict with:
        blur_origin_x, blur_origin_y  — light source screen position (TextureSpaceBlurOrigin)
        bloom_tint                    — (R, G, B) bloom colour (BloomTint)
        bloom_threshold               — minimum luminance for bloom (BloomThreshold)
        bloom_scale                   — overall bloom intensity (BloomScale)
        bloom_max_brightness          — firefly clamp for the bloom (BloomMaxBrightness)
        occlusion_depth_range         — depth range for occlusion mask
        occlusion_mask_darkness       — floor opacity for the occlusion mask
        aspect_ratio_x, aspect_ratio_y— viewport aspect ratio pair

    鲁迅式：光柱参数是光的名片——屏幕坐标、颜色、强度、景深范围，
    缺少任何一项，光柱便失去了身份。
    """
    cx = bbox.get("x", 0) + bbox.get("w", 100) / 2.0
    cy = bbox.get("y", 0) + bbox.get("h", 50)  / 2.0

    # TextureSpaceBlurOrigin: normalised screen position
    blur_origin_x = cx / max(viewport_w, 1.0)
    blur_origin_y = cy / max(viewport_h, 1.0)

    # Aspect ratio pair (mirrors AspectRatioAndInvAspectRatio.XY / ZW)
    ar = viewport_w / max(viewport_h, 1.0)

    # Species-driven bloom colour
    _BLOOM_TINTS = {
        "cil-eye":  (0.85, 0.90, 1.00),   # cool indigo glow
        "cil-bolt": (1.00, 0.70, 0.20),   # warm amber burst
        "cil-loop": (1.00, 0.80, 0.30),   # golden cycle
    }
    bloom_tint = _BLOOM_TINTS.get("default", (0.9, 0.9, 0.9))
    for sp_key, tint in _BLOOM_TINTS.items():
        if sp_key in cell_id:
            bloom_tint = tint
            break

    return {
        "blur_origin_x":        round(blur_origin_x, 4),
        "blur_origin_y":        round(blur_origin_y, 4),
        "bloom_tint":           bloom_tint,
        "bloom_threshold":      0.0,   # BloomThreshold (0 = all pixels contribute)
        "bloom_scale":          0.25,  # BloomScale
        "bloom_max_brightness": 10.0,  # BloomMaxBrightness (firefly clamp)
        "occlusion_depth_range": _LS_OCCLUSION_RANGE,
        "occlusion_mask_darkness": _LS_OCCLUSION_DARK,
        "aspect_ratio_x":       round(ar, 4),
        "aspect_ratio_y":       round(1.0 / max(ar, 1e-6), 4),
    }





def render_cell_light_shafts(
    cell_id:    str,
    species:    str,
    bbox:       dict,
    viewport_w: float = 1200.0,
    viewport_h: float = 900.0,
) -> str:
    """
    Top-level light shaft renderer — mirrors the dispatch in
    FDeferredShadingSceneRenderer::RenderLightShaftBloom() /
    RenderLightShaftOcclusion().

    Checks whether light shafts should render for this cell/species; if so,
    emits both the bloom filter and the occlusion mask SVG fragments.

    Returns a concatenated SVG string, or empty string if not applicable.

    鲁迅式：光柱渲染是条件艺术——条件不成立，一个字也不写；
    条件成立，则两种技术都要上，缺一不可。
    """
    if not should_render_light_shafts(species):
        return ""
    if not should_render_light_shafts_for_cell(cell_id, bbox, viewport_w, viewport_h):
        return ""

    params   = get_cell_light_shaft_params(cell_id, bbox, viewport_w, viewport_h)
    bloom    = AstroCellLightShaftBloom(params, cell_id, bbox, species)
    occ      = AstroCellLightShaftOcclusion(params, cell_id, bbox)

    bloom_svg = bloom.emit_svg()
    occ_svg   = occ.emit_svg()

    parts = []
    if bloom_svg:
        parts.append(bloom_svg)
    if occ_svg:
        parts.append(occ_svg)

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] ShadowRendering → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/ShadowRendering.cpp
#
# 鲁迅曾言：「真正的勇士，敢于直面惨淡的阴影；
# 然而阴影的深浅，不过是光源距离与偏置的函数。」
# 深度偏置是影子的谎言许可证——允许一点点自欺，换来不互相遮挡的太平。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   CVarCSMShadowDepthBias / CVarCSMShadowSlopeScaleDepthBias
#       → _CSM_DEPTH_BIAS / _CSM_SLOPE_BIAS
#   CVarShadowTransitionScale → _SHADOW_TRANSITION_SCALE
#   CVarCSMShadowReceiverBias → _CSM_RECEIVER_BIAS
#   CVarFilterMethod (0=PCF, 1=PCSS) → _SHADOW_FILTER_METHOD
#   GStencilOptimization / GShadowStencilCulling
#       → _STENCIL_OPTIMIZATION / _STENCIL_CULLING
#
# AstroCellShadowProjection → per-cell depth-bias CSM projection
# AstroCellShadowRenderer   → epoch-level shadow dispatch
# ═══════════════════════════════════════════════════════════════════════════════

_CSM_DEPTH_BIAS:            float = 10.0
_CSM_SLOPE_BIAS:            float = 3.0
_PER_OBJECT_DIR_DEPTH_BIAS: float = 10.0
_PER_OBJECT_DIR_SLOPE_BIAS: float = 3.0
_CSM_RECEIVER_BIAS:         float = 0.9
_POINT_LIGHT_DEPTH_BIAS:    float = 0.02
_POINT_LIGHT_SLOPE_BIAS:    float = 3.0
_RECT_LIGHT_DEPTH_BIAS:     float = 0.025
_RECT_LIGHT_SLOPE_BIAS:     float = 2.5
_RECT_LIGHT_RECEIVER_BIAS:  float = 0.3
_SPOT_LIGHT_DEPTH_BIAS:     float = 3.0
_SPOT_LIGHT_SLOPE_BIAS:     float = 3.0
_SPOT_LIGHT_RECEIVER_BIAS:  float = 0.5
_SHADOW_TRANSITION_SCALE:   float = 60.0
_SPOT_TRANSITION_SCALE:     float = 60.0
_SHADOW_FILTER_METHOD:      int   = 0
_STENCIL_OPTIMIZATION:      bool  = True
_STENCIL_CULLING:           bool  = True
_PCSS_MAX_KERNEL_RADIUS:    float = 40.0
_ENABLE_MODULATED_SELF_SHADOW: bool = False


