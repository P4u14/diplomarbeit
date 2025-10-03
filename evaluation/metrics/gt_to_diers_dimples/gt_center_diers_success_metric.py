from evaluation.metrics.base_metric import Metric


class GTCenterDiersSuccessMetric(Metric):
    def __init__(self, tolerance=3):
        """Initialize metric with a given tolerance distance in [mm]."""
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return f'GT Center DIERS Success (tol={self.tolerance}mm)'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Check if both left and right segment center errors are within the tolerance distance."""
        if image_metadata['pixel_size_mm'] is not None:
            pixel_size_mm = image_metadata['pixel_size_mm']
            allowed_distance_px = self.tolerance / pixel_size_mm
        else:
            allowed_distance_px = self.tolerance

        return int(computed_metric_results['GT Center DIERS Error Left [pixel]'] <= allowed_distance_px and
                computed_metric_results['GT Center DIERS Error Right [pixel]'] <= allowed_distance_px)
