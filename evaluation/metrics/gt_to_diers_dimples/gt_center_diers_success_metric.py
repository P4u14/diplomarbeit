from evaluation.metrics.base_metric import Metric


class GTCenterDiersSuccessMetric(Metric):
    """
    Checks if both left and right DIERS segment center errors (ground truth) are within a specified tolerance (in mm).
    This metric is useful for evaluating whether the ground truth segment centers match the DIERS reference spatially.
    """
    def __init__(self, tolerance=3):
        """
        Initializes the metric with a given tolerance distance in millimeters.
        Parameters:
            tolerance (float): Allowed spatial deviation in mm.
        """
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output, including the tolerance.
        Returns:
            str: The name with tolerance, e.g. 'GT Center DIERS Success (tol=3mm)'.
        """
        return f'GT Center DIERS Success (tol={self.tolerance}mm)'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """
        Checks if both left and right DIERS segment center errors (ground truth) are within the tolerance distance.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics (must include 'GT Center DIERS Error Left [pixel]' and 'GT Center DIERS Error Right [pixel]').
            image_metadata (dict): Dictionary with marker positions and pixel size.
        Returns:
            int: 1 if both errors are within tolerance, 0 otherwise.
        """
        if image_metadata['pixel_size_mm'] is not None:
            pixel_size_mm = image_metadata['pixel_size_mm']
            allowed_distance_px = self.tolerance / pixel_size_mm
        else:
            allowed_distance_px = self.tolerance

        return int(computed_metric_results['GT Center DIERS Error Left [pixel]'] <= allowed_distance_px and
                computed_metric_results['GT Center DIERS Error Right [pixel]'] <= allowed_distance_px)
