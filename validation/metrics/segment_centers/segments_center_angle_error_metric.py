import numpy as np

from validation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric


class SegmentsCenterAngleErrorMetric(SegmentsCenterErrorMetric):
    @property
    def name(self) -> str:
        return 'Segments Center Angle Error [degrees]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute angle distance between the gt and predicted centroids for the left and right side of the back."""
        gt_center_left = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'], side='left')
        pred_center_left = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'], side='left')

        gt_center_right = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'], side='right')
        pred_center_right = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'], side='right')

        # If both centers are missing on one side, return 0.0
        if gt_center_left is None and pred_center_left is None or gt_center_right is None and pred_center_right is None:
            return 0.0
        # If one center is missing on either side, return NaN
        if None in (gt_center_left, pred_center_left, gt_center_right, pred_center_right):
            return float('nan')

        dx_gt = gt_center_right[0] - gt_center_left[0]
        dy_gt = gt_center_right[1] - gt_center_left[1]
        angle_gt = np.degrees(np.arctan2(dy_gt, dx_gt))

        dx_pred = pred_center_right[0] - pred_center_left[0]
        dy_pred = pred_center_right[1] - pred_center_left[1]
        angle_pred = np.degrees(np.arctan2(dy_pred, dx_pred))

        angle_error = angle_pred - angle_gt
        angle_error = (angle_error + 180) % 360 - 180  # Normalize to [-180, 180]
        return abs(angle_error)
