import numpy as np

from evaluation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric


class GTCenterAngleDiersErrorMetric(SegmentsCenterErrorMetric):
    """
    Computes the absolute angle error (in degrees) between the DIERS measurement and the ground truth centroids
    for the left and right side of the back. This metric quantifies the angular deviation between
    the DIERS reference and the ground truth segment centers.
    """

    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'GT Center Angle DIERS Error [degrees]'.
        """
        return 'GT Center Angle DIERS Error [degrees]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """
        Calculates the absolute angle distance between the DIERS measurement and ground truth centroids for the left and right side.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask (not used).
            computed_metric_results (dict): Dictionary containing previously computed metrics.
            image_metadata (dict): Dictionary with marker positions and pixel size.
        Returns:
            float: Absolute angle error in degrees.
        """
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
