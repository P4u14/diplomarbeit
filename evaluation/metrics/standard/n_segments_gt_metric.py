from skimage.measure import label

from evaluation.metrics.base_metric import Metric

class NSegmentsGTMetric(Metric):
    """
    Computes the number of segments in the ground truth segmentation mask.
    Useful for quantifying the amount of annotated anatomical structures or regions.
    """
    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'N Segments GT'.
        """
        return 'N Segments GT'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """
        Calculates the number of segments in the ground truth mask.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics.
            image_metadata (dict, optional): Additional image metadata (not used).
        Returns:
            int: Number of ground truth segments.
        """
        n_gt_segments = int(label(gt).max())

        return n_gt_segments
