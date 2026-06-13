"""Shadow setup, shadow depth rendering, and deferred shading renderer."""
from channels.rendering.shadow.shadow import (
    AstroCellShadowInfo,
    AstroCellShadowDepthPassParams,
    AstroCellShadowDepthRenderer,
    AstroCellGBuffer,
    AstroCellDeferredShadingRenderer,
    build_whole_scene_shadow_caster,
    run_deferred_shading_pipeline,
)
