"""
frame_consistency.py — Cross-Frame Layer Consistency Checker
===============================================================
Replaces the incorrect variance > avg * 0.5 check in
layer_separator.py (line 607-609) with coefficient of variation (CV).

From Megatron-LM's `schedules.py` forward_backward_pipelining_without_interleaving():
  Each microbatch's forward+backward is tracked precisely via
  input_tensors / output_tensors lists.  Cross-rank consistency
  uses average_losses_across_data_parallel_group() for proper
  statistical aggregation — not raw variance.

Milestone: M104
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyReport:
    """Result of cross-frame layer consistency analysis."""
    consistent: bool
    severity: Literal["ok", "warning", "error"]
    mean_layers: float
    std_layers: float
    cv: float                         # coefficient of variation = std / mean
    outlier_frames: List[int]         # frame indices with outlier layer counts
    total_frames: int = 0
    layer_counts: List[int] = field(default_factory=list)
    message: str = ""

    def to_dict(self):
        return {
            "consistent": self.consistent,
            "severity": self.severity,
            "mean_layers": round(self.mean_layers, 2),
            "std_layers": round(self.std_layers, 2),
            "cv": round(self.cv, 4),
            "outlier_frames": self.outlier_frames,
            "total_frames": self.total_frames,
            "message": self.message,
        }


class FrameConsistencyChecker:
    """Statistically correct cross-frame consistency analysis.

    Uses coefficient of variation (CV = std/mean) instead of the
    incorrect variance > avg * 0.5 check.  CV is dimensionless and
    works correctly regardless of the mean layer count.

    Thresholds:
      CV < 0.15 → "ok"      (std is <15% of mean)
      CV < 0.40 → "warning" (noticeable variation but usable)
      CV >= 0.40 → "error"  (layers are unstable across frames)
    """

    def __init__(
        self,
        cv_ok: float = 0.15,
        cv_warn: float = 0.40,
        outlier_sigma: float = 2.0,
    ):
        self.cv_ok = cv_ok
        self.cv_warn = cv_warn
        self.outlier_sigma = outlier_sigma

    def check(self, layer_counts: List[int]) -> ConsistencyReport:
        """Analyze layer count consistency across frames.

        Parameters
        ----------
        layer_counts : list of ints
            Number of layers extracted per frame.

        Returns
        -------
        ConsistencyReport
        """
        n = len(layer_counts)

        if n < 2:
            return ConsistencyReport(
                consistent=True,
                severity="ok",
                mean_layers=layer_counts[0] if layer_counts else 0,
                std_layers=0.0,
                cv=0.0,
                outlier_frames=[],
                total_frames=n,
                layer_counts=layer_counts,
                message="Single frame — consistency check not applicable",
            )

        mean = sum(layer_counts) / n
        variance = sum((c - mean) ** 2 for c in layer_counts) / n
        std = math.sqrt(variance)

        # Coefficient of variation (handle zero mean)
        cv = std / mean if mean > 0 else 0.0

        # Detect outlier frames (> outlier_sigma standard deviations from mean)
        outliers = []
        if std > 0:
            for i, count in enumerate(layer_counts):
                z = abs(count - mean) / std
                if z > self.outlier_sigma:
                    outliers.append(i)

        # Severity
        if cv < self.cv_ok:
            severity: Literal["ok", "warning", "error"] = "ok"
            consistent = True
            msg = f"Consistent: CV={cv:.3f} (<{self.cv_ok})"
        elif cv < self.cv_warn:
            severity = "warning"
            consistent = True
            msg = (
                f"Mild variation: CV={cv:.3f} "
                f"(mean={mean:.1f}±{std:.1f} layers)"
            )
        else:
            severity = "error"
            consistent = False
            msg = (
                f"Inconsistent: CV={cv:.3f} (>{self.cv_warn}), "
                f"mean={mean:.1f}±{std:.1f} layers, "
                f"{len(outliers)} outlier frame(s): {outliers}"
            )

        logger.info("FrameConsistency: %s", msg)

        return ConsistencyReport(
            consistent=consistent,
            severity=severity,
            mean_layers=mean,
            std_layers=std,
            cv=cv,
            outlier_frames=outliers,
            total_frames=n,
            layer_counts=layer_counts,
            message=msg,
        )
