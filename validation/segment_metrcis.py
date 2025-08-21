from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentMetrics:
    dice: float
    precision: float
    recall: float
    n_gt_segments: int
    n_pred_segments: int
    dimples_center_left_deviation: tuple[int, int]
    dimples_center_right_deviation: tuple[int, int]