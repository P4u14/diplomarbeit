import numpy as np

from evaluation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric


class SegmentsCenterErrorLeftMetric(SegmentsCenterErrorMetric):
    """
    Computes the absolute distance (in pixels) between the ground truth and predicted centroid for all segments on the left side of the back.
    This metric quantifies the spatial deviation between predicted and true segment centers on the left.
    """

    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'Segments Center Error Left [pixel]'.
        """
        return 'Segments Center Error Left [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """
        Calculates the absolute distance between the ground truth and predicted centroid for all segments on the left side.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics.
            image_metadata (dict): Dictionary with marker positions and pixel size.
        Returns:
            float: Euclidean distance in pixels.
        """
        gt_center = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'], side='left')
        pred_center = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'], side='left')

        if gt_center is None and pred_center is None:
            return 0.0
        if gt_center is None or pred_center is None:
            return float('inf')

        euklid_dist = np.linalg.norm(np.array(gt_center) - np.array(pred_center))
        return euklid_dist
