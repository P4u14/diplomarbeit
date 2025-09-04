import csv
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.measure import label
from tqdm import tqdm
from collections import defaultdict

from validation.segment_metrcis import SegmentMetrics


class Validator:

    def __init__(self, ground_truth_dir, output_dir):
        self.ground_truth_dir = ground_truth_dir
        self.ground_truths = self.load_masks(ground_truth_dir)
        self.output_dir = output_dir

    def validate(self, predictions_dir):
        ground_truths = self.ground_truths
        predictions = self.load_masks(predictions_dir)
        # # store current prediction directory for loading original images
        # current_predictions_dir = predictions_dir

        # Collect per-file rows and group metrics by dataset
        rows: list[list] = []
        metrics_by_set: dict[str, list[SegmentMetrics]] = defaultdict(list)

        for file_name in tqdm(ground_truths.keys(), desc=f'Validating predictions for {predictions_dir}'):
            if file_name not in predictions.keys():
                print(f"Ground truth {file_name} does not have a corresponding prediction.")
                continue

            dataset = self.parse_dataset(file_name)
            gt = ground_truths[file_name]
            prediction = predictions[file_name]

            metrics = self.compute_metrics(file_name, gt, prediction)

            # store row and group metrics
            rows.append([
                dataset, file_name,
                metrics.tp, metrics.fp, metrics.fn,
                metrics.dice, metrics.precision, metrics.recall,
                metrics.n_gt_segments, metrics.n_pred_segments,
                metrics.n_gt_segments_left, metrics.n_pred_segments_left,
                metrics.n_gt_segments_right, metrics.n_pred_segments_right,
                metrics.center_gt_left, metrics.center_pred_left,
                metrics.center_gt_right, metrics.center_pred_right,
                metrics.center_pred_success,
                metrics.center_angle_error, metrics.center_angle_error_abs,
                metrics.center_angle_success
            ])
            metrics_by_set[dataset].append(metrics)

        # prepare output directories
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = os.path.basename(predictions_dir)
        all_csv = output_dir / f"{run_name}_all.csv"
        mean_csv = output_dir / f"{run_name}_mean.csv"

        with open(all_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [
                'Dataset', 'File Name',
                'TP', 'FP', 'FN',
                'Dice', 'Precision', 'Recall',
                'N GT Segments', 'N Pred Segments',
                'N GT Segments Left', 'N Pred Segments Left',
                'N GT Segments Right', 'N of Pred Segments Right',
                'Center GT Left', 'Center Pred Left',
                'Center GT Right', 'Center Pred Right',
                'Center Pred Success',
                'Center Angle Error', 'Center Angle Error Abs',
                'Center Angle Success'
            ]
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Validation results saved to {all_csv}")

        # Write mean metrics grouped by dataset and overall
        with open(mean_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header_mean = [
                'Dataset',
                'Mean TP', 'Mean FP', 'Mean FN',
                'Mean Dice', 'Mean Precision', 'Mean Recall',
                'Mean N GT Segments', 'Mean N Pred Segments',
                'Mean N GT Segments Left', 'Mean N Pred Segments Left',
                'Mean N GT Segments Right', 'Mean N Pred Segments Right',
                'Mean Center GT Left (x,y)', 'Mean Center Pred Left (x,y)',
                'Mean Center GT Right (x,y)', 'Mean Center Pred Right (x,y)',
                'Mean Center Pred Success',
                'Mean Center Angle Error', 'Mean Center Angle Error Abs',
                'Mean Center Angle Success'
            ]
            writer.writerow(header_mean)
            # per-dataset means
            for dataset, mlist in metrics_by_set.items():
                writer.writerow([
                    dataset,
                    _safe_nanmean([m.tp for m in mlist]),
                    _safe_nanmean([m.fp for m in mlist]),
                    _safe_nanmean([m.fn for m in mlist]),
                    _safe_nanmean([m.dice for m in mlist]),
                    _safe_nanmean([m.precision for m in mlist]),
                    _safe_nanmean([m.recall for m in mlist]),
                    _safe_nanmean([m.n_gt_segments for m in mlist]),
                    _safe_nanmean([m.n_pred_segments for m in mlist]),
                    _safe_nanmean([m.n_gt_segments_left for m in mlist]),
                    _safe_nanmean([m.n_pred_segments_left for m in mlist]),
                    _safe_nanmean([m.n_gt_segments_right for m in mlist]),
                    _safe_nanmean([m.n_pred_segments_right for m in mlist]),
                    (
                        _safe_nanmean([m.center_gt_left[0] for m in mlist if m.center_gt_left is not None]),
                        _safe_nanmean([m.center_gt_left[1] for m in mlist if m.center_gt_left is not None])
                    ),
                    (
                        _safe_nanmean([m.center_pred_left[0] for m in mlist if m.center_pred_left is not None]),
                        _safe_nanmean([m.center_pred_left[1] for m in mlist if m.center_pred_left is not None])
                    ),
                    (
                        _safe_nanmean([m.center_gt_right[0] for m in mlist if m.center_gt_right is not None]),
                        _safe_nanmean([m.center_gt_right[1] for m in mlist if m.center_gt_right is not None])
                    ),
                    (
                        _safe_nanmean([m.center_pred_right[0] for m in mlist if m.center_pred_right is not None]),
                        _safe_nanmean([m.center_pred_right[1] for m in mlist if m.center_pred_right is not None])
                    ),
                    _safe_nanmean([m.center_pred_success for m in mlist if m.center_pred_success is not None]),
                    _safe_nanmean([m.center_angle_error for m in mlist if m.center_angle_error is not None]),
                    _safe_nanmean([m.center_angle_error_abs for m in mlist if m.center_angle_error_abs is not None]),
                    _safe_nanmean([m.center_angle_success for m in mlist if m.center_angle_success is not None])
                ])
            # overall mean
            all_metrics = [m for lst in metrics_by_set.values() for m in lst]
            writer.writerow([
                'All Datasets',
                _safe_nanmean([m.tp for m in all_metrics]),
                _safe_nanmean([m.fp for m in all_metrics]),
                _safe_nanmean([m.fn for m in all_metrics]),
                _safe_nanmean([m.dice for m in all_metrics]),
                _safe_nanmean([m.precision for m in all_metrics]),
                _safe_nanmean([m.recall for m in all_metrics]),
                _safe_nanmean([m.n_gt_segments for m in all_metrics]),
                _safe_nanmean([m.n_pred_segments for m in all_metrics]),
                _safe_nanmean([m.n_gt_segments_left for m in all_metrics]),
                _safe_nanmean([m.n_pred_segments_left for m in all_metrics]),
                _safe_nanmean([m.n_gt_segments_right for m in all_metrics]),
                _safe_nanmean([m.n_pred_segments_right for m in all_metrics]),
                (
                    _safe_nanmean([m.center_gt_left[0] for m in all_metrics if m.center_gt_left is not None]),
                    _safe_nanmean([m.center_gt_left[1] for m in all_metrics if m.center_gt_left is not None])
                ),
                (
                    _safe_nanmean([m.center_pred_left[0] for m in all_metrics if m.center_pred_left is not None]),
                    _safe_nanmean([m.center_pred_left[1] for m in all_metrics if m.center_pred_left is not None])
                ),
                (
                    _safe_nanmean([m.center_gt_right[0] for m in all_metrics if m.center_gt_right is not None]),
                    _safe_nanmean([m.center_gt_right[1] for m in all_metrics if m.center_gt_right is not None])
                ),
                (
                    _safe_nanmean([m.center_pred_right[0] for m in all_metrics if m.center_pred_right is not None]),
                    _safe_nanmean([m.center_pred_right[1] for m in all_metrics if m.center_pred_right is not None])
                ),
                _safe_nanmean([m.center_pred_success for m in all_metrics if m.center_pred_success is not None]),
                _safe_nanmean([m.center_angle_error for m in all_metrics if m.center_angle_error is not None]),
                _safe_nanmean([m.center_angle_error_abs for m in all_metrics if m.center_angle_error_abs is not None]),
                _safe_nanmean([m.center_angle_success for m in all_metrics if m.center_angle_success is not None])
            ])
        print(f"Mean validation results saved to {mean_csv}")

    @staticmethod
    def parse_dataset(file_name):
        prefix = file_name.split('_')[0]
        return prefix


    @staticmethod
    def load_masks(segmentations_dir):
        segmentations = {}
        for file in os.listdir(segmentations_dir):
            if file.endswith(".png") and "-mask" in file:
                segmentations[file] = io.imread(os.path.join(segmentations_dir, file))
        return segmentations

    def compute_metrics(self, file_name, gt, pred):
        # Binary masks
        gt_mask = gt > 0
        pred_mask = pred > 0

        # Load markers
        vp, dm = self.load_markers(file_name)

        # Initialize object to hold metrics
        metrics = SegmentMetrics()

        # True Positives, False Positives, False Negatives
        metrics.tp = np.logical_and(gt_mask, pred_mask).sum()
        metrics.fp = np.logical_and(~gt_mask, pred_mask).sum()
        metrics.fn = np.logical_and(gt_mask, ~pred_mask).sum()

        # Dice Coefficient
        metrics.dice = self.compute_dice(gt_mask, pred_mask, metrics)

        # Precision
        metrics.precision = self.compute_precision(metrics)

        # Recall
        metrics.recall = self.compute_recall(metrics)

        # Total number of segments
        metrics.n_gt_segments = int(label(gt_mask).max())
        metrics.n_pred_segments = int(label(pred_mask).max())
        metrics.n_gt_segments_left = self.compute_n_segments(gt_mask, vp, dm, side='left')
        metrics.n_pred_segments_left = self.compute_n_segments(pred_mask, vp, dm, side='left')
        metrics.n_gt_segments_right = self.compute_n_segments(gt_mask, vp, dm, side='right')
        metrics.n_pred_segments_right = self.compute_n_segments(pred_mask, vp, dm, side='right')

        # Center of left and right dimples
        metrics.center_gt_left = self.compute_center(gt_mask, vp, dm, side='left')
        metrics.center_pred_left = self.compute_center(pred_mask, vp, dm, side='left')
        metrics.center_gt_right = self.compute_center(gt_mask, vp, dm, side='right')
        metrics.center_pred_right = self.compute_center(pred_mask, vp, dm, side='right')
        # Check prediction success after centers are set
        metrics.center_pred_success = self.compute_center_pred_success(metrics)
        metrics.center_angle_error = self.compute_center_angle_error(metrics, absolute=False)
        metrics.center_angle_error_abs = self.compute_center_angle_error(metrics, absolute=True)
        metrics.center_angle_success = self.compute_center_angle_success(metrics)

        # # display original mask with splitting line overlay in pink
        # # load original images (same name without '-mask')
        # from pathlib import Path
        # base_name = file_name.replace('-mask', '')
        # orig_gt_path = Path(self.ground_truth_dir) / base_name
        # orig_pred_path = Path(self.current_predictions_dir) / base_name
        # try:
        #     orig_gt_img = io.imread(orig_gt_path)
        #     self.visualize_middle_line(orig_gt_img, vp, dm, f"{file_name} Ground Truth Split on Original")
        # except Exception:
        #     pass
        # try:
        #     orig_pred_img = io.imread(orig_pred_path)
        #     self.visualize_middle_line(orig_pred_img, vp, dm, f"{file_name} Pred Split on Original")
        # except Exception:
        #     pass

        return metrics

    @staticmethod
    def compute_dice(gt_mask, pred_mask, metrics):
        if gt_mask.sum() + pred_mask.sum() == 0:
            dice = 1.0
        else:
            dice = (2. * metrics.tp) / (gt_mask.sum() + pred_mask.sum())
        return dice

    @staticmethod
    def compute_precision(metrics):
        if metrics.tp + metrics.fp == 0:
            precision = 1
        else:
            precision = metrics.tp / (metrics.tp + metrics.fp)
        return precision

    @staticmethod
    def compute_recall(metrics):
        if metrics.tp + metrics.fn == 0:
            recall = 1
        else:
            recall = metrics.tp / (metrics.tp + metrics.fn)
        return recall

    def load_markers(self, file_name, markers_file="data/Info_sheets/Markerpositionen.csv"):
        img_number = self.extract_image_number(file_name)

        with open(markers_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                if row.get('BildID', '').startswith(str(img_number)):
                    vp = (int(row['X_VP']) / 10, int(row['Y_VP']) / 10)
                    dm = (int(row['X_DM']) / 10, int(row['Y_DM']) / 10)
                    return vp, dm

        pat_idx = self.parse_patient_index_from_image_path(file_name)
        with open('data/Info_Sheets/All_Data_Renamed_overview.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('PatientsLikeMe') == pat_idx:
                    measure_id = row['DIERS_Mess-ID']

        with open(markers_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('MessID', '') == measure_id:
                    vp = (int(row['X_VP'] / 10), int(row['Y_VP']) / 10)
                    dm = (int(row['X_DM'] / 10), int(row['Y_DM']) / 10)
                    return vp, dm

        return None, None

    @staticmethod
    def compute_n_segments(mask, vp, dm, side):
        """Count labeled segments in the left or right half of a binary mask."""
        # prepare 2D binary mask
        m = mask.astype(bool)
        if m.ndim == 3:
            m = m[..., 0]
        h, w = m.shape
        # coordinate grid
        x = np.arange(w)[None, :]
        y = np.arange(h)[:, None]
        # split by marker line or center
        if vp is not None and dm is not None:
            x1, y1 = vp
            x2, y2 = dm
            s = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            half = (s < 0) if side == 'left' else (s >= 0)
        else:
            mid = w // 2
            half = (x < mid) if side == 'left' else (x >= mid)
        region = m & half
        return int(label(region).max())

    @staticmethod
    def compute_center(mask, vp, dm, side):
        """Compute centroid of mask in left or right half region."""
        # prepare binary mask
        m = mask.astype(bool)
        if m.ndim == 3:
            m = m[..., 0]
        h, w = m.shape
        # coordinate grid
        x = np.arange(w)[None, :]
        y = np.arange(h)[:, None]
        # define half by marker line or center
        if vp is not None and dm is not None:
            x1, y1 = vp
            x2, y2 = dm
            s = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            half = (s < 0) if side == 'left' else (s >= 0)
        else:
            mid = w // 2
            half = (x < mid) if side == 'left' else (x >= mid)
        region = m & half
        # find coordinates of pixels in region
        ys, xs = np.where(region)
        if xs.size == 0:
            return None
        # compute centroid
        x_center = float(xs.mean())
        y_center = float(ys.mean())
        return x_center, y_center

    @staticmethod
    def parse_patient_index_from_image_path(image_path):
        pattern = re.compile(r'^[^_]+_([^_]+(?:_\d+)+)(?=_\d{9,})')
        match = pattern.search(os.path.basename(image_path))
        if match:
            return match.group(1)
        return None

    @staticmethod
    def extract_image_number(file_name):
        match = re.search(r'_(\d+)-mask\.Gauss\.png$', file_name)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def visualize_mask(image: np.ndarray, title: str):
        """Display the original mask image without any splitting lines or axes."""
        fig, ax = plt.subplots()
        # display mask or full-color image
        if image.ndim == 3 and image.shape[2] >= 3:
            ax.imshow(image, aspect='equal', interpolation='nearest')
        else:
            vis = image[...,0] if image.ndim == 3 else image
            ax.imshow(vis, cmap='gray', aspect='equal', interpolation='nearest')
        ax.axis('off')
        ax.set_title(title)
        plt.show()

    @staticmethod
    def visualize_middle_line(image: np.ndarray, vp: Optional[Tuple[int, int]], dm: Optional[Tuple[int, int]], title: str):
        """Display the mask image with a middle splitting line."""
        fig, ax = plt.subplots()
        # if image has 3 or more channels, show in RGB, else grayscale
        if image.ndim == 3 and image.shape[2] >= 3:
            ax.imshow(image, aspect='equal', interpolation='nearest')
        else:
            vis = image[...,0] if image.ndim == 3 else image
            ax.imshow(vis, cmap='gray', aspect='equal', interpolation='nearest')

        if vp is not None and dm is not None:
            # Draw the splitting line in the middle of the two markers
            xs = [vp[0], dm[0]]
            ys = [vp[1], dm[1]]
            ax.plot(xs, ys, color='magenta', linewidth=2)

        ax.axis('off')
        ax.set_title(title)
        plt.show()

    @staticmethod
    def compute_center_angle_error(metrics, absolute=False):
        """Compute signed or absolute angle difference (in degrees) between GT and predicted dimple-center line."""
        gt_l = metrics.center_gt_left; gt_r = metrics.center_gt_right
        pr_l = metrics.center_pred_left; pr_r = metrics.center_pred_right
        # ensure all centers are available
        if None in (gt_l, gt_r, pr_l, pr_r):
            return None
        # compute line angles
        dx_gt = gt_r[0] - gt_l[0]; dy_gt = gt_r[1] - gt_l[1]
        dx_pr = pr_r[0] - pr_l[0]; dy_pr = pr_r[1] - pr_l[1]
        angle_gt = np.degrees(np.arctan2(dy_gt, dx_gt))
        angle_pr = np.degrees(np.arctan2(dy_pr, dx_pr))
        # error and normalization to [-180,180]
        err = angle_pr - angle_gt
        err = (err + 180) % 360 - 180
        return abs(err) if absolute else err

    @staticmethod
    def compute_center_angle_success(metrics, threshold=3.0):
        """Return 1 if absolute angle error is below threshold degrees, else 0."""
        err = metrics.center_angle_error_abs
        if err is None:
            return None
        return int(err < threshold)

    @staticmethod
    def compute_center_pred_success(metrics, threshold=5.0):
        """Return 1 if both left and right center predictions are correct (both None or within threshold px of GT)."""
        gl, pl = metrics.center_gt_left, metrics.center_pred_left
        gr, pr = metrics.center_gt_right, metrics.center_pred_right
        def ok(gt, pr):
            if gt is None and pr is None:
                return True
            if gt is not None and pr is not None:
                # distance within 10 pixels
                return np.hypot(gt[0] - pr[0], gt[1] - pr[1]) <= threshold
            return False
        return int(ok(gl, pl) and ok(gr, pr))

def _safe_nanmean(arr):
    if len(arr) == 0:
        return None
    mean_val = np.nanmean(arr)
    return float(mean_val)
