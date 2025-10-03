from skimage.measure import label

from evaluation.metrics.base_metric import Metric

class NSegmentsPredMetric(Metric):
    """
    Computes the number of segments in the predicted segmentation mask.
    Useful for quantifying the amount of predicted anatomical structures or regions.
    """
    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'N Segments Pred'.
        """
        return 'N Segments Pred'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """
        Calculates the number of segments in the prediction mask.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics.
            image_metadata (dict, optional): Additional image metadata (not used).
        Returns:
            int: Number of predicted segments.
        """
        n_pred_segments = int(label(pred).max())

        return n_pred_segments
