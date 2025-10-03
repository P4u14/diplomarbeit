from evaluation.metrics.base_metric import Metric

class PrecisionMetric(Metric):
    """
    Computes the precision for segmentation: TP / (TP + FP).
    Precision quantifies the proportion of positive identifications that were actually correct.
    """
    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'Precision'.
        """
        return 'Precision'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """
        Calculates the precision: TP / (TP + FP). Returns NaN if denominator is zero.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics (must include 'TP' and 'FP').
            image_metadata (dict, optional): Additional image metadata (not used).
        Returns:
            float: Precision value or NaN if denominator is zero.
        """
        tp = computed_metric_results['TP']
        fp = computed_metric_results['FP']
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float('nan')
