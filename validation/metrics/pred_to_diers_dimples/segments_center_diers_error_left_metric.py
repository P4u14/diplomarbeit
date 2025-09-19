import numpy as np

from validation.metrics.segments_center_error_metric import SegmentsCenterErrorMetric


class SegmentsCenterDiersErrorLeftMetric(SegmentsCenterErrorMetric):
    @property
    def name(self) -> str:
        return 'Segments Center DIERS Error Left [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute distance between the DIERS measurement and predicted centroid for all segments on the left side of the back."""
        diers_center = image_metadata['dl_diers']
        pred_center = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'], side='left')

        euklid_dist = np.linalg.norm(np.array(diers_center) - np.array(pred_center))
        return euklid_dist
