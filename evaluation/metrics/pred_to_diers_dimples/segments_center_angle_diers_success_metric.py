from evaluation.metrics.base_metric import Metric


class SegmentsCenterAngleDiersSuccessMetric(Metric):
    """
    Checks if the DIERS center angle error is within a specified tolerance (in degrees).
    This metric is useful for evaluating whether the predicted segment orientation matches the DIERS reference within a given threshold.
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
            str: The name with tolerance, e.g. 'Segments Center Angle DIERS Success (tol=4.2°)'.
        """
        return f'Segments Center Angle DIERS Success (tol={self.tolerance}\u00b0)'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """
        Checks if the DIERS center angle error is within the tolerance degrees.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics (must include 'Segments Center Angle DIERS Error [degrees]').
            image_metadata (dict): Dictionary with marker positions and pixel size.
        Returns:
            int: 1 if within tolerance, 0 otherwise.
        """
        return int(computed_metric_results['Segments Center Angle DIERS Error [degrees]'] <= self.tolerance)
