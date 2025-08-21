from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentMetrics:
    dice: float
    precision: float
    recall: float
    n_gt_segments: int
    n_pred_segments: int