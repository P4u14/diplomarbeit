from validation.metrics.base_metric import Metric


class SegmentsCenterAngleSuccessMetric(Metric):
    def __init__(self, tolerance=4.2):
        """Initialize metric with a given tolerance distance in [degrees]."""
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return f'Segments Center Angle Success (tol={self.tolerance}°)'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Check if center angle error is within the tolerance degrees."""
        return int(computed_metric_results['Segments Center Angle Error [degrees]'] <= self.tolerance)
