import numpy as np

from validation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric


class GTCenterDiersErrorRightMetric(SegmentsCenterErrorMetric):
    @property
    def name(self) -> str:
        return 'GT Center DIERS Error Right [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute distance between the DIERS measurement and gt centroid for all segments on the right side of the back."""
        diers_center = image_metadata['dr_diers']
        gt_center = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'], side='right')

        if gt_center is None and diers_center is None:
            return 0.0
        if gt_center is None or diers_center is None:
            return float('inf')

        euklid_dist = np.linalg.norm(np.array(diers_center) - np.array(gt_center))
        return euklid_dist
