from evaluation.metrics.base_metric import Metric

class DiceMetric(Metric):
    @property
    def name(self) -> str:
        return 'Dice'

    def compute(self, gt, pred, computed_metric_results, image_metadata=None):
        """Compute Dice coefficient: 2*TP / (sum of sizes)"""
        tp = computed_metric_results['TP']
        total = gt.sum() + pred.sum()
        if total == 0:
            return 1.0
        return 2 * tp / total
