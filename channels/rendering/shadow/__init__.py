"""Shadow setup, depth rendering, and deferred shading classes."""
from .shadow import (
    AstroCellShadowInfo,
    build_whole_scene_shadow_caster,
    AstroCellShadowDepthPassParams,
    AstroCellShadowDepthRenderer,
    AstroCellGBuffer,
    AstroCellDeferredShadingRenderer,
    run_deferred_shading_pipeline,
)

__all__ = [
    "AstroCellShadowInfo",
    "build_whole_scene_shadow_caster",
    "AstroCellShadowDepthPassParams",
    "AstroCellShadowDepthRenderer",
    "AstroCellGBuffer",
    "AstroCellDeferredShadingRenderer",
    "run_deferred_shading_pipeline",
]
