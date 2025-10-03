from evaluation.metrics.base_metric import Metric


class GTCenterAngleDiersSuccessMetric(Metric):
    """
    Checks if the DIERS center angle error (ground truth) is within a specified tolerance (in degrees).
    This metric is useful for evaluating whether the ground truth segment orientation matches the DIERS reference within a given threshold.
    """
    def __init__(self, tolerance=4.2):
        """
        Initializes the metric with a given tolerance distance in degrees.
        Parameters:
            tolerance (float): Allowed angular deviation in degrees.
        """
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output, including the tolerance.
        Returns:
            str: The name with tolerance, e.g. 'GT Center Angle DIERS Success (tol=4.2°)'.
        """
        return f'GT Center Angle DIERS Success (tol={self.tolerance}°)'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """
        Checks if the DIERS center angle error (ground truth) is within the tolerance degrees.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics (must include 'GT Center Angle DIERS Error [degrees]').
            image_metadata (dict): Dictionary with marker positions and pixel size.
        Returns:
            int: 1 if within tolerance, 0 otherwise.
        """
        return int(computed_metric_results['GT Center Angle DIERS Error [degrees]'] <= self.tolerance)
