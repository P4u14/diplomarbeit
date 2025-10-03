import numpy as np

from evaluation.metrics.base_metric import Metric


class SegmentsCenterErrorMetric(Metric):
    """
    Computes the absolute distance (in pixels) between the ground truth and predicted centroid for all segments.
    This metric quantifies the spatial deviation between predicted and true segment centers.
    """

    @property
    def name(self) -> str:
        """
        Returns the name of the metric for CSV output.
        Returns:
            str: The name 'Segments Center Error [pixel]'.
        """
        return 'Segments Center Error [pixel]'

    def compute(self, gt, pred, computed_metric_results, image_metadata):
        """
        Calculates the absolute distance between the ground truth and predicted centroid for all segments.
        Parameters:
            gt (np.ndarray): Ground truth binary mask.
            pred (np.ndarray): Predicted binary mask.
            computed_metric_results (dict): Dictionary containing previously computed metrics.
            image_metadata (dict): Dictionary with marker positions and pixel size.
        Returns:
            float: Euclidean distance in pixels.
        """
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
        """
        Computes the centroid of a binary mask, optionally restricted to a side defined by the vanishing point (vp)
        and the marker line (dm).
        Parameters:
            mask (np.ndarray): Binary mask.
            vp (tuple): Vanishing point coordinates (x, y).
            dm (tuple): Marker line coordinates (x, y).
            side (str, optional): Side of the mask to compute the center for ('left' or 'right').
                                  If None, the center of the entire mask is computed.
        Returns:
            tuple: Centroid coordinates (x, y) or None if no pixels are found.
        """
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

