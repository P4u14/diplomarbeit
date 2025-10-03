from evaluation.metrics.base_metric import Metric


class GTCenterAngleDiersSuccessMetric(Metric):
    def __init__(self, tolerance=4.2):
        """Initialize metric with a given tolerance distance in [degrees]."""
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return f'GT Center Angle DIERS Success (tol={self.tolerance}°)'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Check if center angle error is within the tolerance degrees."""
        return int(computed_metric_results['GT Center Angle DIERS Error [degrees]'] <= self.tolerance)
