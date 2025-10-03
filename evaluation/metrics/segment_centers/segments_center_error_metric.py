import numpy as np

from evaluation.metrics.base_metric import Metric


class SegmentsCenterErrorMetric(Metric):
    @property
    def name(self) -> str:
        return 'Segments Center Error [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """Compute the absolute distance between the gt and predicted centroid for all segments."""
        gt_center = self.compute_center(gt, image_metadata['vp'], image_metadata['dm'])
        pred_center = self.compute_center(pred, image_metadata['vp'], image_metadata['dm'])

        if gt_center is None and pred_center is None:
            return 0.0
        if gt_center is None or pred_center is None:
            return float('inf')

        euklid_dist = np.linalg.norm(np.array(gt_center) - np.array(pred_center))
        return euklid_dist

    @staticmethod
    def compute_center(mask, vp, dm, side=None):
        # prepare binary mask
        if mask.ndim == 3:
            mask = mask[..., 0]
        h, w = mask.shape

        if side is None:
            # If no side is specified, compute center of entire mask
            ys, xs = np.where(mask)
        else:
            # Create coordinate grid
            x = np.arange(w)[None, :]
            y = np.arange(h)[:, None]

            # Define separation line between left and right half of the back by marker line
            x1, y1 = vp
            x2, y2 = dm
            s = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            half = (s < 0) if side == 'left' else (s >= 0)
            region = mask & half

            # Find coordinates of pixels in correct side of the back
            ys, xs = np.where(region)

        # If no pixels found, return None
        if xs.size == 0:
            return None

        # Compute centroid
        x_center = float(xs.mean())
        y_center = float(ys.mean())
        return x_center, y_center