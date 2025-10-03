import numpy as np

from evaluation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric


class SegmentsCenterAngleDiersErrorMetric(SegmentsCenterErrorMetric):
    @property
    def name(self) -> str:
        return 'Segments Center Angle DIERS Error [degrees]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute angle distance between the DIERS measurement and predicted centroids for the left and right side of the back."""
        diers_center_left = image_metadata['dl_diers']
        pred_center_left = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'], side='left')

        diers_center_right = image_metadata['dr_diers']
        pred_center_right = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'], side='right')

        # If both centers are missing on one side, return 0.0
        if diers_center_left is None and pred_center_left is None or diers_center_right is None and pred_center_right is None:
            return 0.0
        # If one center is missing on either side, return NaN
        if None in (diers_center_left, pred_center_left, diers_center_right, pred_center_right):
            return float('nan')

        dx_gt = diers_center_right[0] - diers_center_left[0]
        dy_gt = diers_center_right[1] - diers_center_left[1]
        angle_gt = np.degrees(np.arctan2(dy_gt, dx_gt))

        dx_pred = pred_center_right[0] - pred_center_left[0]
        dy_pred = pred_center_right[1] - pred_center_left[1]
        angle_pred = np.degrees(np.arctan2(dy_pred, dx_pred))

        angle_error = angle_pred - angle_gt
        angle_error = (angle_error + 180) % 360 - 180  # Normalize to [-180, 180]
        return abs(angle_error)
