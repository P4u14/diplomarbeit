import numpy as np

from validation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric


class GTCenterAngleDiersErrorMetric(SegmentsCenterErrorMetric):
    @property
    def name(self) -> str:
        return 'GT Center Angle DIERS Error [degrees]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute angle distance between the DIERS measurement and GT centroids for the left and right side of the back."""
        diers_center_left = image_metadata['dl_diers']
        gt_center_left = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'], side='left')

        diers_center_right = image_metadata['dr_diers']
        gt_center_right = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'], side='right')

        # If both centers are missing on one side, return 0.0
        if diers_center_left is None and gt_center_left is None or diers_center_right is None and gt_center_right is None:
            return 0.0
        # If one center is missing on either side, return NaN
        if None in (diers_center_left, gt_center_left, diers_center_right, gt_center_right):
            return float('nan')

        dx_gt = diers_center_right[0] - diers_center_left[0]
        dy_gt = diers_center_right[1] - diers_center_left[1]
        angle_gt = np.degrees(np.arctan2(dy_gt, dx_gt))

        dx_pred = gt_center_right[0] - gt_center_left[0]
        dy_pred = gt_center_right[1] - gt_center_left[1]
        angle_pred = np.degrees(np.arctan2(dy_pred, dx_pred))

        angle_error = angle_pred - angle_gt
        angle_error = (angle_error + 180) % 360 - 180  # Normalize to [-180, 180]
        return abs(angle_error)
