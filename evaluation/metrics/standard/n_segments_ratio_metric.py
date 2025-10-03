from skimage.measure import label

from evaluation.metrics.base_metric import Metric

class NSegmentsRatioMetric(Metric):
    @property
    def name(self) -> str:
        return 'N Segments Ratio (pred/gt)'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """Compute the ratio of the number of segments in prediction to ground truth."""
        n_gt_segments = int(label(gt).max())
        n_pred_segments = int(label(pred).max())

        if n_gt_segments == 0 and n_pred_segments == 0:
            return 1.0

        if n_gt_segments == 0:
            return float('inf')

        return n_pred_segments / n_gt_segments
