from validation.metrics.base_metric import Metric


class SegmentsCenterAngleErrorMetric(Metric):
    @property
    def name(self) -> str:
        return 'Segments Center Angle Error [degrees]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute angle distance between the gt and predicted centroids for the left and right side of the back."""


