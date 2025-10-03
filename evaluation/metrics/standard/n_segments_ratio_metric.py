from skimage.measure import label

from evaluation.metrics.base_metric import Metric

class NSegmentsRatioMetric(Metric):
    """
    Computes the ratio of the number of predicted segments to ground truth segments in a segmentation mask.
    Useful for evaluating over- or under-segmentation.
    """
    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'N Segments Ratio (pred/gt)'.
        """
        return 'N Segments Ratio (pred/gt)'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """
        Calculates the ratio of the number of segments in prediction to ground truth.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics.
            image_metadata (dict, optional): Additional image metadata (not used).
        Returns:
            float: Ratio of predicted to ground truth segments.
        """
        n_gt_segments = int(label(gt).max())
        n_pred_segments = int(label(pred).max())

        if n_gt_segments == 0 and n_pred_segments == 0:
            return 1.0

        if n_gt_segments == 0:
            return float('inf')

        return n_pred_segments / n_gt_segments
