from evaluation.metrics.base_metric import Metric

class RecallMetric(Metric):
    """
    Computes the recall (sensitivity) for segmentation: TP / (TP + FN).
    Recall quantifies the proportion of actual positives that were correctly identified.
    """
    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'Recall'.
        """
        return 'Recall'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """
        Calculates the recall (sensitivity): TP / (TP + FN). Returns NaN if denominator is zero.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics (must include 'TP' and 'FN').
            image_metadata (dict, optional): Additional image metadata (not used).
        Returns:
            float: Recall value or NaN if denominator is zero.
        """
        tp = computed_metric_results['TP']
        fn = computed_metric_results['FN']
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float('nan')
