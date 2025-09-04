from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SegmentMetrics:
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    dice: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    n_gt_segments: Optional[int] = None
    n_pred_segments: Optional[int] = None
    n_gt_segments_left: Optional[int] = None
    n_pred_segments_left: Optional[int] = None
    n_gt_segments_right: Optional[int] = None
    n_pred_segments_right: Optional[int] = None
    center_gt_left: Optional[Tuple[float, float]] = None
    center_gt_right: Optional[Tuple[float, float]] = None
    center_pred_left: Optional[Tuple[float, float]] = None
    center_pred_right: Optional[Tuple[float, float]] = None
    center_pred_success: Optional[int] = None
    center_angle_error: Optional[float] = None
    center_angle_error_abs: Optional[float] = None
    center_angle_success: Optional[int] = None