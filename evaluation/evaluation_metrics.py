from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EvaluationMetrics:
    """
    Data class for storing evaluation metrics for segmentation tasks.
    Each attribute represents a specific metric or result from the evaluation process.

    Attributes:
        tp (Optional[int]): True positives count.
        fp (Optional[int]): False positives count.
        fn (Optional[int]): False negatives count.
        dice (Optional[float]): Dice coefficient.
        precision (Optional[float]): Precision score.
        recall (Optional[float]): Recall score.
        n_gt_segments (Optional[int]): Number of ground truth segments.
        n_pred_segments (Optional[int]): Number of predicted segments.
        n_gt_segments_left (Optional[int]): Number of ground truth segments on the left side.
        n_pred_segments_left (Optional[int]): Number of predicted segments on the left side.
        n_gt_segments_right (Optional[int]): Number of ground truth segments on the right side.
        n_pred_segments_right (Optional[int]): Number of predicted segments on the right side.
        n_segments_success (Optional[int]): Number of successfully matched segments.
        center_gt_left (Optional[Tuple[float, float]]): Center coordinates of ground truth segment (left).
        center_gt_right (Optional[Tuple[float, float]]): Center coordinates of ground truth segment (right).
        center_pred_left (Optional[Tuple[float, float]]): Center coordinates of predicted segment (left).
        center_pred_right (Optional[Tuple[float, float]]): Center coordinates of predicted segment (right).
        center_pred_success (Optional[int]): Number of successful center predictions.
        center_diers_left (Optional[Tuple[float, float]]): Center coordinates from Diers system (left).
        center_diers_right (Optional[Tuple[float, float]]): Center coordinates from Diers system (right).
        center_diers_success (Optional[int]): Number of successful Diers center matches.
        center_angle_error (Optional[float]): Error in center angle (in degrees or radians).
        center_angle_error_abs (Optional[float]): Absolute error in center angle.
        center_angle_success (Optional[int]): Number of successful center angle predictions.
        center_angle_diers_error (Optional[float]): Error in Diers center angle.
        center_angle_diers_error_abs (Optional[float]): Absolute error in Diers center angle.
        center_angle_diers_success (Optional[int]): Number of successful Diers center angle predictions.
    """
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
    n_segments_success: Optional[int] = None
    center_gt_left: Optional[Tuple[float, float]] = None
    center_gt_right: Optional[Tuple[float, float]] = None
    center_pred_left: Optional[Tuple[float, float]] = None
    center_pred_right: Optional[Tuple[float, float]] = None
    center_pred_success: Optional[int] = None
    center_diers_left: Optional[Tuple[float, float]] = None
    center_diers_right: Optional[Tuple[float, float]] = None
    center_diers_success: Optional[int] = None
    center_angle_error: Optional[float] = None
    center_angle_error_abs: Optional[float] = None
    center_angle_success: Optional[int] = None
    center_angle_diers_error: Optional[float] = None
    center_angle_diers_error_abs: Optional[float] = None
    center_angle_diers_success: Optional[int] = None