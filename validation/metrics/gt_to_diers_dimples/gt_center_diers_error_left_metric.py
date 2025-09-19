import numpy as np

from validation.metrics.segments_center_error_metric import SegmentsCenterErrorMetric


class GTCenterDiersErrorLeftMetric(SegmentsCenterErrorMetric):
    @property
    def name(self) -> str:
        return 'GT Center DIERS Error Left [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute distance between the DIERS measurement and gt centroid for all segments on the left side of the back."""
        diers_center = image_metadata['dl_diers']
        gt_center = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'], side='left')

        euklid_dist = np.linalg.norm(np.array(diers_center) - np.array(gt_center))
        return euklid_dist
