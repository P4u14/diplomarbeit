import numpy as np

from validation.metrics.segments_center_error_metric import SegmentsCenterErrorMetric


class SegmentsCenterErrorLeftMetric(SegmentsCenterErrorMetric):
    @property
    def name(self) -> str:
        return 'Segments Center Error Left [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute distance between the gt and predicted centroid for all segments on the left side of the back."""
        gt_center = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'], side='left')
        pred_center = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'], side='left')

        euklid_dist = np.linalg.norm(np.array(gt_center) - np.array(pred_center))
        return euklid_dist
