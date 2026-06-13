import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)





def reset_perf_counters() -> None:
    """重置所有性能计数器 — 镜像帧间 GPU stat 清零。"""
    for k in _ASTRO_CELL_PERF_COUNTERS:
        _ASTRO_CELL_PERF_COUNTERS[k] = 0









def increment_perf_counter(name: str, delta: int = 1) -> None:
    """递增指定性能计数器；键不存在时静默创建。"""
    _ASTRO_CELL_PERF_COUNTERS[name] = _ASTRO_CELL_PERF_COUNTERS.get(name, 0) + delta









def use_vector_render_path() -> bool:
    """判断是否启用矢量批渲染路径 — 镜像 UseMeshShader() / UsePrimitiveShader()。

    在 UE5 中，Mesh Shader / Primitive Shader 路径在支持 Tier-1 Mesh Shader
    的平台上激活，以减少 draw call 开销。Astro 的对应逻辑：当 PixiJS 渲染器
    支持 WebGL2 Instanced Mesh（``ASTRO_VECTOR_RENDER=1`` 环境变量）时返回
    True，否则退化为逐元素 SVG 路径（镜像 VertexShader fallback）。
    """
    import os
    return os.environ.get("ASTRO_VECTOR_RENDER", "0") == "1"




