import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)





class AstroCellFeedbackManager:
    """缓冲区溢出反馈管理器 — 镜像 Nanite::FFeedbackManager。

    维护四条高水位线状态（nodes / candidate_clusters / visible_clusters /
    visible_assembly_parts），并在超容时向 stderr 写入警告。亦可跟踪 species
    材质性能警告（对应 ``CVarEmitMaterialPerformanceWarnings``）。

    Attributes
    ----------
    high_water_marks : dict[str, int]
        各缓冲区的历史峰值（镜像 FBufferState::HighWaterMark）。
    material_warnings : dict[str, float]
        species 名 → 最后一次出现的时间戳（镜像 MaterialWarningItems）。
    """

    def __init__(self) -> None:
        self.high_water_marks: dict[str, int] = {
            "nodes":                0,
            "candidate_clusters":   0,
            "visible_clusters":     0,
            "visible_assembly_parts": 0,
        }
        self._latest_overflow: dict[str, float] = {}
        self.material_warnings: dict[str, float] = {}

    def _update_state(self, key: str, peak: int, capacity: int) -> bool:
        """更新单条缓冲区高水位线，溢出时记录时间戳并返回 True。

        镜像 ``FFeedbackManager::FBufferState::Update``。
        """
        import time
        new_hwm = peak > self.high_water_marks[key]
        if peak > capacity:
            self._latest_overflow[key] = time.monotonic()
        self.high_water_marks[key] = max(self.high_water_marks[key], peak)
        return new_hwm and peak > capacity

    def tick(self, peak_nodes: int, peak_candidate_clusters: int,
             peak_visible_clusters: int, peak_visible_assembly_parts: int) -> None:
        """每帧调用，镜像 GPU Message 回调触发时机。

        将四个峰值与 :class:`AstroCellGlobalResources` 的容量上限比对；
        一旦突破历史高水位且发生溢出，向 stderr 输出警告，与 UE5 的
        ``UE_LOGF(LogRenderer, Warning, …)`` 语义对应。
        """
        checks = [
            ("nodes",                  peak_nodes,                  AstroCellGlobalResources.get_max_nodes(),
             "node buffer", "ASTRO_CELL_MAX_NODES"),
            ("candidate_clusters",     peak_candidate_clusters,     AstroCellGlobalResources.get_max_candidate_clusters(),
             "candidate cluster buffer", "ASTRO_CELL_MAX_CANDIDATE_CLUSTERS"),
            ("visible_clusters",       peak_visible_clusters,       AstroCellGlobalResources.get_max_visible_clusters(),
             "visible cluster buffer", "ASTRO_CELL_MAX_VISIBLE_CLUSTERS"),
            ("visible_assembly_parts", peak_visible_assembly_parts, AstroCellGlobalResources.get_max_visible_assembly_parts(),
             "visible assembly part buffer", "ASTRO_CELL_MAX_VISIBLE_ASSEMBLY_PARTS"),
        ]
        for key, peak, cap, label, cvar in checks:
            if self._update_state(key, peak, cap):
                print(
                    f"[AstroCellFeedback] WARNING: {label} overflow detected. "
                    f"New high-water mark is {self.high_water_marks[key]} / {cap}. "
                    f"Increase {cvar} to prevent visual artifacts.",
                    file=__import__("sys").stderr,
                )

    def report_material_perf_warning(self, species_name: str) -> None:
        """报告 species 材质性能警告 — 镜像 ReportMaterialPerformanceWarning。

        对同一 species 采用 5 秒冷却窗口限流，避免日志洪泛。
        仅当 :data:`ASTRO_EMIT_MATERIAL_PERF_WARNINGS` 为 True 时输出。
        """
        if not ASTRO_EMIT_MATERIAL_PERF_WARNINGS:
            return
        import time
        now = time.monotonic()
        last = self.material_warnings.get(species_name, 0.0)
        if now - last > 5.0:
            self.material_warnings[species_name] = now
            print(
                f"[AstroCellFeedback] Performance Warning: "
                f"Programmable Astro species '{species_name}' uses PDO or is Masked!",
                file=__import__("sys").stderr,
            )


#: 进程级单例 — 镜像 FFeedbackManager 内嵌于 FGlobalResources 的生命周期。
_astro_cell_feedback_manager: AstroCellFeedbackManager = AstroCellFeedbackManager()









def get_feedback_manager() -> AstroCellFeedbackManager:
    """返回进程级反馈管理器单例。"""
    return _astro_cell_feedback_manager


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellStreamExport → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteStreamOut.cpp
#
# NaniteStreamOut 将 GPU 端可见 cluster 的顶点/索引数据流出到 CPU 可访问缓冲区，
# 用于物理碰撞、几何体烘焙等离线用途。Astro 等价实现：将可见 cell 的 SVG 片段
# 与 bbox 元数据序列化到磁盘（或内存缓冲区），供外部工具（布局优化器、碰撞检测
# 服务）消费。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FStreamOutQueueParameters   → AstroCellStreamExportQueue（优先级队列）
#   FInitQueueCS                → AstroCellStreamExporter.init_queue()
#   FNaniteStreamOutTraversalCS → AstroCellStreamExporter.traverse()
#   FAllocateRangesCS           → AstroCellStreamExporter.allocate_ranges()
#   FNaniteStreamOutCS          → AstroCellStreamExporter.stream_out()
#   GNaniteStreamOutCacheTraversalData
#       → ASTRO_STREAM_OUT_CACHE_TRAVERSAL（模块级布尔常量）
# ═══════════════════════════════════════════════════════════════════════════════

#: 镜像 GNaniteStreamOutCacheTraversalData — 为 True 时在 count pass 中缓存
#: traversal 中间结果，stream-out pass 可跳过重复遍历。
ASTRO_STREAM_OUT_CACHE_TRAVERSAL: bool = True




