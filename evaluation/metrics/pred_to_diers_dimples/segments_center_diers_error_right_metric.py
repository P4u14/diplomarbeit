import numpy as np

from evaluation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric


class SegmentsCenterDiersErrorRightMetric(SegmentsCenterErrorMetric):
    """
    Computes the absolute distance (in pixels) between the DIERS measurement and the predicted centroid for all segments on the right side of the back.
    This metric quantifies the spatial deviation between the DIERS reference and the predicted segment center (right).
    """

    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'Segments Center DIERS Error Right [pixel]'.
        """
        return 'Segments Center DIERS Error Right [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """
        Calculates the absolute distance between the DIERS measurement and predicted centroid for all segments on the right side.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics.
            image_metadata (dict): Dictionary with marker positions and pixel size.
        Returns:
            float: Euclidean distance in pixels.
        """
        diers_center = image_metadata['dr_diers']
        pred_center = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'], side='right')

        if diers_center is None and pred_center is None:
            return 0.0
        if diers_center is None or pred_center is None:
            return float('inf')

        euklid_dist = np.linalg.norm(np.array(diers_center) - np.array(pred_center))
        return euklid_dist
