import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)









def get_primitive_id_buffer_pool() -> AstroCellPrimitiveIdBufferPool:
    """Return the process-level primitive ID buffer pool singleton."""
    return _ASTRO_PRIMITIVE_ID_BUFFER_POOL


# =============================================================================
# [MeshPassProcessor] AstroCellPipelineStateId + AstroCellShaderBindings
# =============================================================================

# PSO freeze flag — mirrors FGraphicsMinimalPipelineStateId::bIsIdTableFrozen.
_pso_table_frozen: bool = False

# Persistent PSO table — mirrors FGraphicsMinimalPipelineStateId::PersistentIdTable.
# Key: (species, blend_mode, pass_name); Value: integer PSO id.
_pipeline_state_table: dict = {}
_pipeline_state_next_id: int = 0








