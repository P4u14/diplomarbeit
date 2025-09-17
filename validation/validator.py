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

from validation.evaluation_metrics import EvaluationMetrics


class Validator:

    def __init__(self, ground_truth_dir, output_dir):
        self.ground_truth_dir = ground_truth_dir
        self.ground_truths = self.load_masks(ground_truth_dir)
        self.output_dir = output_dir
        self.vp_dm_distances = self.load_vp_dm_distances()

    def validate(self, predictions_dir):
        ground_truths = self.ground_truths
        predictions = self.load_masks(predictions_dir)


        # Collect per-file rows and group metrics by dataset
        rows: list[list] = []
        metrics_by_set: dict[str, list[EvaluationMetrics]] = defaultdict(list)
        metrics_by_sick: dict[str, list[EvaluationMetrics]] = defaultdict(list)

        # load mapping from Patientenindex to Krank value
        sick_map = self.load_patient_sick_map()

        # iterate over all ground truth files and validate predictions
        for file_name in tqdm(ground_truths.keys(), desc=f'Validating predictions for {predictions_dir}'):
            if file_name not in predictions.keys():
                print(f"Ground truth {file_name} does not have a corresponding prediction.")
                continue

            dataset = self.parse_dataset(file_name)
            gt = ground_truths[file_name]
            prediction = predictions[file_name]

            metrics = self.compute_metrics(file_name, gt, prediction)

            # determine sick status from patient index
            pat_idx = self.parse_patient_index_from_image_path(file_name)
            sick = sick_map.get(pat_idx)

            # collect all data in a row
            rows.append([
                dataset, file_name, sick,
                metrics.tp, metrics.fp, metrics.fn,
                metrics.dice, metrics.precision, metrics.recall,
                metrics.n_gt_segments, metrics.n_pred_segments,
                metrics.n_gt_segments_left, metrics.n_pred_segments_left,
                metrics.n_gt_segments_right, metrics.n_pred_segments_right,
                metrics.n_segments_success,
                metrics.center_gt_left, metrics.center_pred_left,
                metrics.center_gt_right, metrics.center_pred_right,
                metrics.center_pred_success,
                metrics.center_diers_left, metrics.center_diers_right,
                metrics.center_diers_success,
                metrics.center_angle_error, metrics.center_angle_error_abs,
                metrics.center_angle_success,
                metrics.center_angle_diers_error, metrics.center_angle_diers_error_abs,
                metrics.center_angle_diers_success
            ])

            # group by dataset
            metrics_by_set[dataset].append(metrics)

            # group by sick status: 1.0 sick, 0.0 healthy
            if sick == 1.0:
                metrics_by_sick['Sick'].append(metrics)
            elif sick == 0.0:
                metrics_by_sick['Healthy'].append(metrics)

        # prepare output directories
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = os.path.basename(predictions_dir)
        all_csv = output_dir / f"{run_name}_all.csv"
        mean_csv = output_dir / f"{run_name}_mean.csv"

        # write all per-file metrics
        with open(all_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [
                'Dataset', 'File Name', 'Sick',
                'TP', 'FP', 'FN',
                'Dice', 'Precision', 'Recall',
                'N GT Segments', 'N Pred Segments',
                'N GT Segments Left', 'N Pred Segments Left',
                'N GT Segments Right', 'N of Pred Segments Right',
                'N Segments Success',
                'Center GT Left', 'Center Pred Left',
                'Center GT Right', 'Center Pred Right',
                'Center Pred Success',
                'Center Diers Left', 'Center Diers Right',
                'Center Diers Success',
                'Center Angle Error', 'Center Angle Error Abs',
                'Center Angle Success',
                'Center Angle Diers Error', 'Center Angle Diers Error Abs',
                'Center Angle Diers Success'
            ]
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Validation results saved to {all_csv}")

        # Write mean metrics grouped by dataset, by sick status and overall
        with open(mean_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header_mean = [
                'Dataset',
                'TP', 'FP', 'FN',
                'Dice', 'Precision', 'Recall',
                'N GT Segments', 'N Pred Segments',
                'N GT Segments Left', 'N Pred Segments Left',
                'N GT Segments Right', 'N of Pred Segments Right',
                'N Segments Success',
                'Center GT Left', 'Center Pred Left',
                'Center GT Right', 'Center Pred Right',
                'Center Pred Success',
                'Center Diers Left', 'Center Diers Right',
                'Center Diers Success',
                'Center Angle Error', 'Center Angle Error Abs',
                'Center Angle Success',
                'Center Angle Diers Error', 'Center Angle Diers Error Abs',
                'Center Angle Diers Success'
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
                    _safe_nanmean([m.n_segments_success for m in mlist if m.n_segments_success is not None]),
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
                    (
                        _safe_nanmean([m.center_diers_left[0] for m in mlist if m.center_diers_left is not None]),
                        _safe_nanmean([m.center_diers_left[1] for m in mlist if m.center_diers_left is not None])
                    ),
                    (
                        _safe_nanmean([m.center_diers_right[0] for m in mlist if m.center_diers_right is not None]),
                        _safe_nanmean([m.center_diers_right[1] for m in mlist if m.center_diers_right is not None])
                    ),
                    _safe_nanmean([m.center_diers_success for m in mlist if m.center_diers_success is not None]),
                    _safe_nanmean([m.center_angle_error for m in mlist if m.center_angle_error is not None]),
                    _safe_nanmean([m.center_angle_error_abs for m in mlist if m.center_angle_error_abs is not None]),
                    _safe_nanmean([m.center_angle_success for m in mlist if m.center_angle_success is not None]),
                    _safe_nanmean([m.center_angle_diers_error for m in mlist if m.center_angle_diers_error is not None]),
                    _safe_nanmean([m.center_angle_diers_error_abs for m in mlist if m.center_angle_diers_error_abs is not None]),
                    _safe_nanmean([m.center_angle_diers_success for m in mlist if m.center_angle_diers_success is not None])
                ])
            # Sick and Healthy means
            for status, mlist in [('Sick', metrics_by_sick['Sick']), ('Healthy', metrics_by_sick['Healthy'])]:
                writer.writerow([
                    status,
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
                    _safe_nanmean([m.n_segments_success for m in mlist if m.n_segments_success is not None]),
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
                    (
                        _safe_nanmean([m.center_diers_left[0] for m in mlist if m.center_diers_left is not None]),
                        _safe_nanmean([m.center_diers_left[1] for m in mlist if m.center_diers_left is not None])
                    ),
                    (
                        _safe_nanmean([m.center_diers_right[0] for m in mlist if m.center_diers_right is not None]),
                        _safe_nanmean([m.center_diers_right[1] for m in mlist if m.center_diers_right is not None])
                    ),
                    _safe_nanmean([m.center_diers_success for m in mlist if m.center_diers_success is not None]),
                    _safe_nanmean([m.center_angle_error for m in mlist if m.center_angle_error is not None]),
                    _safe_nanmean([m.center_angle_error_abs for m in mlist if m.center_angle_error_abs is not None]),
                    _safe_nanmean([m.center_angle_success for m in mlist if m.center_angle_success is not None]),
                    _safe_nanmean([m.center_angle_diers_error for m in mlist if m.center_angle_diers_error is not None]),
                    _safe_nanmean([m.center_angle_diers_error_abs for m in mlist if m.center_angle_diers_error_abs is not None]),
                    _safe_nanmean([m.center_angle_diers_success for m in mlist if m.center_angle_diers_success is not None])
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
                _safe_nanmean([m.n_segments_success for m in all_metrics if m.n_segments_success is not None]),
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
                (
                    _safe_nanmean([m.center_diers_left[0] for m in all_metrics if m.center_diers_left is not None]),
                    _safe_nanmean([m.center_diers_left[1] for m in all_metrics if m.center_diers_left is not None])
                ),
                (
                    _safe_nanmean([m.center_diers_right[0] for m in all_metrics if m.center_diers_right is not None]),
                    _safe_nanmean([m.center_diers_right[1] for m in all_metrics if m.center_diers_right is not None])
                ),
                _safe_nanmean([m.center_diers_success for m in all_metrics if m.center_diers_success is not None]),
                _safe_nanmean([m.center_angle_error for m in all_metrics if m.center_angle_error is not None]),
                _safe_nanmean([m.center_angle_error_abs for m in all_metrics if m.center_angle_error_abs is not None]),
                _safe_nanmean([m.center_angle_success for m in all_metrics if m.center_angle_success is not None]),
                _safe_nanmean([m.center_angle_diers_error for m in all_metrics if m.center_angle_diers_error is not None]),
                _safe_nanmean([m.center_angle_diers_error_abs for m in all_metrics if m.center_angle_diers_error_abs is not None]),
                _safe_nanmean([m.center_angle_diers_success for m in all_metrics if m.center_angle_diers_success is not None])
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
        vp, dm, dl_diers, dr_diers = self.load_markers(file_name)

        # Calc pixel ratio in mm
        pat_idx = self.parse_patient_index_from_image_path(file_name)
        pixel_size = self.compute_distance_per_pixel(pat_idx, vp, dm)

        # Initialize object to hold metrics
        metrics = EvaluationMetrics()

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
        metrics.n_segments_success = self.compute_n_segments_success(gt_mask, pred_mask)

        # Center of left and right dimples
        metrics.center_gt_left = self.compute_center(gt_mask, vp, dm, side='left')
        metrics.center_pred_left = self.compute_center(pred_mask, vp, dm, side='left')
        metrics.center_gt_right = self.compute_center(gt_mask, vp, dm, side='right')
        metrics.center_pred_right = self.compute_center(pred_mask, vp, dm, side='right')
        metrics.center_pred_success = self.compute_center_pred_success(metrics, pixel_size)

        # Compute center distance to diers captured data (!not necessarily equal to gt if gt is annotated differently!)
        metrics.center_diers_left = dl_diers
        metrics.center_diers_right = dr_diers
        metrics.center_diers_success = self.compute_center_pred_success(metrics, pixel_size, compare_to_diers=True)

        # Angle error between center lines
        metrics.center_angle_error = self.compute_center_angle_error(metrics, absolute=False)
        metrics.center_angle_error_abs = self.compute_center_angle_error(metrics, absolute=True)
        metrics.center_angle_success = self.compute_center_angle_success(metrics)

        # Angle error between center lines to diers captured data (!not necessarily equal to gt if gt is annotated differently!)
        metrics.center_angle_diers_error = self.compute_center_angle_error(metrics, absolute=False, compare_to_diers=True)
        metrics.center_angle_diers_error_abs = self.compute_center_angle_error(metrics, absolute=True, compare_to_diers=True)
        metrics.center_angle_diers_success = self.compute_center_angle_success(metrics, compare_to_diers=True)

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

    @staticmethod
    def compute_n_segments_success(gt_mask, pred_mask):
        """Return 1 if every GT segment has at least one pixel in the prediction; Also 1 if no GT segments."""
        # label ground truth components
        labels_gt = label(gt_mask)
        n = labels_gt.max()
        if n == 0:
            return 1
        # check overlap for each segment
        for seg in range(1, n+1):
            if not np.any(pred_mask[labels_gt == seg]):
                return 0
        return 1

    def load_markers(self, file_name, markers_file="data/Info_sheets/Markerpositionen.csv"):
        img_number = self.extract_image_number(file_name)

        with open(markers_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                if row.get('BildID', '').startswith(str(img_number)):
                    vp = (int(row['X_VP']) / 10, int(row['Y_VP']) / 10)
                    dm = (int(row['X_DM']) / 10, int(row['Y_DM']) / 10)
                    dl_diers = (int(row['X_DL']) / 10, int(row['Y_DL']) / 10)
                    dr_diers = (int(row['X_DR']) / 10, int(row['Y_DR']) / 10)
                    return vp, dm, dl_diers, dr_diers

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
                    dl_diers = (int(row['X_DL']) / 10, int(row['Y_DL']) / 10)
                    dr_diers = (int(row['X_DR']) / 10, int(row['Y_DR']) / 10)
                    return vp, dm, dl_diers, dr_diers

        return None, None, None, None

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
    def compute_center_angle_error(metrics, absolute=False, compare_to_diers=False):
        """Compute signed or absolute angle difference (in degrees) between GT and predicted dimple-center line."""
        if compare_to_diers:
            gt_l = metrics.center_diers_left; gt_r = metrics.center_diers_right
            pr_l = metrics.center_pred_left; pr_r = metrics.center_pred_right
        else:
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
    def compute_center_angle_success(metrics, threshold=4.2, compare_to_diers=False):
        """Return 1 if absolute angle error is below threshold degrees, else 0."""
        if compare_to_diers:
            err = metrics.center_angle_diers_error_abs
        else:
            err = metrics.center_angle_error_abs
        if err is None:
            return None
        return int(err < threshold)

    @staticmethod
    def compute_center_pred_success(metrics, pixel_size, threshold=3.0, compare_to_diers=False):
        """Return 1 if both left and right center predictions are correct (both None or within threshold [mm] of GT)."""
        if pixel_size is not None:
            threshold = threshold / pixel_size
        if compare_to_diers:
            gl, pl = metrics.center_diers_left, metrics.center_pred_left
            gr, pr = metrics.center_diers_right, metrics.center_pred_right
        else:
            gl, pl = metrics.center_gt_left, metrics.center_pred_left
            gr, pr = metrics.center_gt_right, metrics.center_pred_right
        def ok(gt, pr):
            if gt is None and pr is None:
                return True
            if gt is not None and pr is not None:
                # distance within threshold pixels
                return np.hypot(gt[0] - pr[0], gt[1] - pr[1]) <= threshold
            return False
        return int(ok(gl, pl) and ok(gr, pr))

    def compute_distance_per_pixel(self, patient_idx, vp, dm):
        """
        Compute millimeters per pixel using known physical distance between VP and DM markers.
        """
        if vp is None or dm is None:
            return None
        # compute pixel distance between markers
        pixel_dist = np.hypot(dm[0] - vp[0], dm[1] - vp[1])
        if pixel_dist == 0:
            return None
        # get physical distance (mm) for this patient
        mm_dist = self.vp_dm_distances.get(patient_idx)
        if mm_dist is None:
            return None
        # mm per pixel
        return mm_dist / pixel_dist

    @staticmethod
    def load_patient_sick_map(file_path="data/Info_Sheets/All_Data_Renamed_overview.csv"):
        """Load the mapping from Patientenindex to Krank value."""
        sick_map = {}
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pat_idx = row.get('Patientenindex')
                sick = row.get('Krank')
                if pat_idx and sick:
                    sick_map[pat_idx] = float(sick)
        return sick_map

    @staticmethod
    def load_vp_dm_distances(file_path="data/Info_Sheets/All_Data_Renamed_overview.csv"):
        vp_dm_distance_map = {}
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pat_idx = row.get('Patientenindex')
                dist_str = row.get('Rumpflänge')
                if pat_idx and dist_str:
                    try:
                        dist_mm = float(dist_str)
                        vp_dm_distance_map[pat_idx] = dist_mm
                    except ValueError:
                        continue
        return vp_dm_distance_map


def _safe_nanmean(arr):
    if len(arr) == 0:
        return None
    mean_val = np.nanmean(arr)
    return float(mean_val)
