from skimage.measure import label

from evaluation.metrics.base_metric import Metric

class NSegmentsGTMetric(Metric):
    @property
    def name(self) -> str:
        return 'N Segments GT'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """Compute the number of segments in the ground truth."""
        n_gt_segments = int(label(gt).max())

        return n_gt_segments
