from skimage.measure import label

from evaluation.metrics.base_metric import Metric

class NSegmentsPredMetric(Metric):
    @property
    def name(self) -> str:
        return 'N Segments Pred'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """Compute the number of segments in the prediction."""
        n_pred_segments = int(label(pred).max())

        return n_pred_segments
