from evaluation.metrics.base_metric import Metric


class SegmentsCenterDiersSuccessMetric(Metric):
    """
    Checks if both left and right DIERS segment center errors are within a specified tolerance (in mm).
    This metric is useful for evaluating whether the predicted segment centers match the DIERS reference spatially.
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
            str: The name with tolerance, e.g. 'Segments Center DIERS Success (tol=3mm)'.
        """
        return f'Segments Center DIERS Success (tol={self.tolerance}mm)'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """
        Checks if both left and right DIERS segment center errors are within the tolerance distance.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics (must include 'Segments Center DIERS Error Left [pixel]' and 'Segments Center DIERS Error Right [pixel]').
            image_metadata (dict): Dictionary with marker positions and pixel size.
        Returns:
            int: 1 if both errors are within tolerance, 0 otherwise.
        """
        if image_metadata['pixel_size_mm'] is not None:
            pixel_size_mm = image_metadata['pixel_size_mm']
            allowed_distance_px = self.tolerance / pixel_size_mm
        else:
            allowed_distance_px = self.tolerance

        return int(computed_metric_results['Segments Center DIERS Error Left [pixel]'] <= allowed_distance_px and
                computed_metric_results['Segments Center DIERS Error Right [pixel]'] <= allowed_distance_px)
