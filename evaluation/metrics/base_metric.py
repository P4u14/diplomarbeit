from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Abstract base class for validation metrics.

    Each metric must have a unique name and implement the compute method,
    which takes ground truth and prediction arrays and returns a numeric score.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric, used as a CSV header."""
        pass

    @abstractmethod
    def compute(self, gt, pred, computed_metric_results, image_metadata) -> float:
        """Compute the metric given ground truth and prediction arrays."""
        pass
