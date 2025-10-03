from evaluation.metrics.base_metric import Metric

class PrecisionMetric(Metric):
    @property
    def name(self) -> str:
        return 'Precision'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """
        Precision: TP / (TP + FP). Returns NaN if denominator is zero.
        """
        tp = computed_metric_results['TP']
        fp = computed_metric_results['FP']
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float('nan')
