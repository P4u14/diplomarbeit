from validation.metrics.base_metric import Metric

class RecallMetric(Metric):
    @property
    def name(self) -> str:
        return 'Recall'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """
        Recall (Sensitivity): TP / (TP + FN). Returns NaN if denominator is zero.
        """
        tp = computed_metric_results['TP']
        fn = computed_metric_results['FN']
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float('nan')
