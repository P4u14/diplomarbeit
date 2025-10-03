import numpy as np

from evaluation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric


class GTCenterDiersErrorRightMetric(SegmentsCenterErrorMetric):
    """
    Computes the absolute distance (in pixels) between the DIERS measurement and the ground truth centroid for all segments on the right side of the back.
    This metric quantifies the spatial deviation between the DIERS reference and the ground truth segment center (right).
    """

    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'GT Center DIERS Error Right [pixel]'.
        """
        return 'GT Center DIERS Error Right [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """
        Calculates the absolute distance between the DIERS measurement and ground truth centroid for all segments on the right side.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics.
            image_metadata (dict): Dictionary with marker positions and pixel size.
        Returns:
            float: Euclidean distance in pixels.
        """
        diers_center = image_metadata['dr_diers']
        gt_center = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'], side='right')

        if gt_center is None and diers_center is None:
            return 0.0
        if gt_center is None or diers_center is None:
            return float('inf')

        euklid_dist = np.linalg.norm(np.array(diers_center) - np.array(gt_center))
        return euklid_dist
