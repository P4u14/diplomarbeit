from evaluation.metrics.base_metric import Metric

class DiceMetric(Metric):
    """
    Computes the Dice coefficient between ground truth and predicted segmentation masks.
    The Dice coefficient measures the overlap between two samples and is commonly used for segmentation evaluation.
    """
    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'Dice'.
        """
        return 'Dice'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """
        Calculates the Dice coefficient: 2*TP / (sum of sizes).
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics (must include 'TP').
            image_metadata (dict, optional): Additional image metadata (not used).
        Returns:
            float: Dice coefficient between 0 and 1.
        """
        tp = computed_metric_results['TP']
        total = gt.sum() + pred.sum()
        if total == 0:
            return 1.0
        return 2 * tp / total
