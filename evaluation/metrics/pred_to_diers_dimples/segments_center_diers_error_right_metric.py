import numpy as np

from evaluation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric


class SegmentsCenterDiersErrorRightMetric(SegmentsCenterErrorMetric):
    @property
    def name(self) -> str:
        return 'Segments Center DIERS Error Right [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute distance between the DIERS measurement and predicted centroid for all segments on the right side of the back."""
        diers_center = image_metadata['dr_diers']
        pred_center = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'], side='right')

        if diers_center is None and pred_center is None:
            return 0.0
        if diers_center is None or pred_center is None:
            return float('inf')

        euklid_dist = np.linalg.norm(np.array(diers_center) - np.array(pred_center))
        return euklid_dist
