import csv
import os
import re
from pathlib import Path

import numpy as np
from skimage import io
from skimage.measure import label
from tqdm import tqdm

from validation.segment_metrcis import SegmentMetrics


class Validator:

    def __init__(self, ground_truth_dir, output_dir):
        self.ground_truth_dir = ground_truth_dir
        self.ground_truths = self.load_masks(ground_truth_dir)
        self.output_dir = output_dir

    def validate(self, predictions_dir):
        ground_truths = self.ground_truths
        predictions = self.load_masks(predictions_dir)

        rows = []
        dice_by_set = {}
        precision_by_set = {}
        recall_by_set = {}
        n_get_mask_by_set = {}
        n_pred_mask_by_set = {}
        dimples_center_left_deviation_set = {}
        dimples_center_right_deviation_set = {}
        dimples_center_left_deviation_set_abs = {}
        dimples_center_right_deviation_set_abs = {}

        for file_name in tqdm(ground_truths.keys(), desc=f'Validating predictions for {predictions_dir}'):
            if file_name not in predictions.keys():
                print(f"Ground truth {file_name} does not have a corresponding prediction.")
                continue

            dataset = self.parse_dataset(file_name)
            gt = ground_truths[file_name]
            prediction = predictions[file_name]

            metrics = self.compute_metrics(file_name, gt, prediction)

            rows.append([
                dataset,
                file_name,
                metrics.dice,
                metrics.precision,
                metrics.recall,
                metrics.n_gt_segments,
                metrics.n_pred_segments,
                metrics.dimples_center_left_deviation,
                metrics.dimples_center_right_deviation
            ])

            dice_by_set.setdefault(dataset, []).append(metrics.dice)
            precision_by_set.setdefault(dataset, []).append(metrics.precision)
            recall_by_set.setdefault(dataset, []).append(metrics.recall)
            n_get_mask_by_set.setdefault(dataset, []).append(metrics.n_gt_segments)
            n_pred_mask_by_set.setdefault(dataset, []).append(metrics.n_pred_segments)
            dimples_center_left_deviation_set.setdefault(dataset, []).append(metrics.dimples_center_left_deviation)
            dimples_center_right_deviation_set.setdefault(dataset, []).append(metrics.dimples_center_right_deviation)
            dimples_center_left_deviation_set_abs.setdefault(dataset, []).append((abs(metrics.dimples_center_left_deviation[0]), abs(metrics.dimples_center_left_deviation[1])))
            dimples_center_right_deviation_set_abs.setdefault(dataset, []).append((abs(metrics.dimples_center_right_deviation[0]), abs(metrics.dimples_center_right_deviation[1])))

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = os.path.basename(predictions_dir)
        all_csv = output_dir / f"{run_name}_all.csv"
        mean_csv = output_dir / f"{run_name}_mean.csv"

        with open(all_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Dataset', 'File Name', 'Dice', 'Precision', 'Recall', 'Number of GT Segments',
                             'Number of Segmentation Segments', 'Dimples Center Left Deviation',
                             'Dimples Center Right Deviation'])
            writer.writerows(rows)
        print(f"Validation results saved to {all_csv}")

        # Mean values
        all_dice = []
        all_precision = []
        all_recall = []
        all_n_gt_mask = []
        all_n_seg_mask = []
        all_deviations_left = []
        all_deviations_right = []
        all_deviations_left_abs = []
        all_deviations_right_abs = []

        for dataset in dice_by_set.keys():
            all_dice.extend(dice_by_set[dataset])
            all_precision.extend(precision_by_set[dataset])
            all_recall.extend(recall_by_set[dataset])
            all_n_gt_mask.extend(n_get_mask_by_set[dataset])
            all_n_seg_mask.extend(n_pred_mask_by_set[dataset])
            all_deviations_left.extend(dimples_center_left_deviation_set[dataset])
            all_deviations_right.extend(dimples_center_right_deviation_set[dataset])
            all_deviations_left_abs.extend(dimples_center_left_deviation_set_abs[dataset])
            all_deviations_right_abs.extend(dimples_center_right_deviation_set_abs[dataset])

        mean_all_dice = np.nanmean(all_dice)
        mean_all_precision = np.nanmean(all_precision)
        mean_all_recall = np.nanmean(all_recall)
        mean_all_n_gt_mask = np.nanmean(all_n_gt_mask)
        mean_all_n_seg_mask = np.nanmean(all_n_seg_mask)
        mean_all_deviations_left = np.nanmean(all_deviations_left)
        mean_all_deviations_right = np.nanmean(all_deviations_right)
        mean_all_deviations_left_abs = np.nanmean(all_deviations_left_abs)
        mean_all_deviations_right_abs = np.nanmean(all_deviations_right_abs)

        with open(mean_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Dataset', 'Mean Dice', 'Mean Precision', 'Mean Recall', 'Mean Number of GT Segments',
                             'Mean Number of Segmentation Segments', 'Mean Dimples Center Left Deviation',
                             'Mean Dimples Center Right Deviation',
                             'Mean Dimples Center Left Deviation Abs', 'Mean Dimples Center Right Deviation Abs'])
            for dataset in dice_by_set.keys():
                writer.writerow([
                                dataset,
                                np.nanmean(dice_by_set[dataset]),
                                np.nanmean(precision_by_set[dataset]),
                                np.nanmean(recall_by_set[dataset]),
                                np.nanmean(n_get_mask_by_set[dataset]),
                                np.nanmean(n_pred_mask_by_set[dataset]),
                                np.nanmean(dimples_center_left_deviation_set[dataset]),
                                np.nanmean(dimples_center_right_deviation_set[dataset]),
                                np.nanmean(dimples_center_left_deviation_set_abs[dataset]),
                                np.nanmean(dimples_center_right_deviation_set_abs[dataset]),
                                ])
            writer.writerow(['All Datasets', mean_all_dice, mean_all_precision, mean_all_recall,
                             mean_all_n_gt_mask, mean_all_n_seg_mask, mean_all_deviations_left,
                             mean_all_deviations_right, mean_all_deviations_left_abs, mean_all_deviations_right_abs])
        print(f"Mean validation results saved to {mean_csv}")

    @staticmethod
    def parse_dataset(file_name):
        prefix_map = {
            "gkge_": "gkge",
            "mBrace_": "mBrace",
            "skolioseKielce_": "skolioseKielce",
            "wip_": "wip",
        }
        for prefix, ds in prefix_map.items():
            if file_name.startswith(prefix):
                return ds
        return "unknown"

    @staticmethod
    def load_masks(segmentations_dir):
        segmentations = {}
        for file in os.listdir(segmentations_dir):
            if file.endswith(".png") and "-mask" in file:
                segmentations[file] = io.imread(os.path.join(segmentations_dir, file))
        return segmentations

    def compute_metrics(self, file_name, gt, pred):
        gt_mask = gt > 0
        pred_mask = pred > 0

        # Dice
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        if gt_mask.sum() + pred_mask.sum() == 0:
            dice = 1.0
        else:
            dice = (2. * intersection) / (gt_mask.sum() + pred_mask.sum())

        # Precision, Recall
        tp = intersection
        fp = np.logical_and(~gt_mask, pred_mask).sum()
        fn = np.logical_and(gt_mask, ~pred_mask).sum()
        if tp + fp == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 1
        else:
            recall = tp / (tp + fn)

        # Number of segments
        n_gt_mask = int(label(gt_mask).max())
        n_seg_mask = int(label(pred_mask).max())

        # centroids
        vp, dm = self.load_markers(file_name)

        if vp is not None and dm is not None:
            # Use markers to split the mask into left and right halves
            dimples_center_left_gt, dimples_center_right_gt = self.half_centroids_from_mask_with_markers(gt_mask, vp,
                                                                                                         dm)
            dimples_center_left_pred, dimples_center_right_pred = self.half_centroids_from_mask_with_markers(pred_mask,
                                                                                                             vp, dm)
        else:
            print(f"\n Warning: No markers found for {file_name}, using default half centroids.")
            dimples_center_left_gt, dimples_center_right_gt = self.half_centroids_from_mask(gt_mask)
            dimples_center_left_pred, dimples_center_right_pred = self.half_centroids_from_mask(pred_mask)

        dimples_center_left_deviation = self.calculate_center_deviation(dimples_center_left_gt,
                                                                        dimples_center_left_pred)
        dimples_center_right_deviation = self.calculate_center_deviation(dimples_center_right_gt,
                                                                         dimples_center_right_pred)

        return SegmentMetrics(
            dice=dice,
            precision=precision,
            recall=recall,
            n_gt_segments=n_gt_mask,
            n_pred_segments=n_seg_mask,
            dimples_center_left_deviation=dimples_center_left_deviation,
            dimples_center_right_deviation=dimples_center_right_deviation
        )

    def load_markers(self, file_name):
        img_number = self.extract_image_number(file_name)
        dataset = self.parse_dataset(file_name)

        vp, dm = self.read_markers_from_file(file_name, img_number, dataset)

        return vp, dm

    def read_markers_from_file(self, file_name, img_number, dataset):
        if (dataset == "gkge") or (dataset == "wip"):
            markers_file = "data/Info_Sheets/Markerpositionen_WiP_GKGE.csv"
        else:
            markers_file = "data/Info_Sheets/Symmetriemarker_SkolioseKielce.csv"

        with open(markers_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                if row.get('BildID', '').startswith(str(img_number)):
                    vp = (int(row['X_VP']), int(row['Y_VP']))
                    dm = (int(row['X_DM']), int(row['Y_DM']))
                    return vp, dm

        pat_idx = self.parse_patient_index_from_image_path(file_name)
        with open('data/Info_Sheets/All_Data_Renamed_overview.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('Patientenindex') == pat_idx:
                    measure_id = row['DIERS_Mess-ID']

        with open(markers_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('MessID', '') == measure_id:
                    vp = (int(row['X_VP']), int(row['Y_VP']))
                    dm = (int(row['X_DM']), int(row['Y_DM']))
                    return vp, dm

        return None, None

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
    def centroid(bin_mask, x, y):
        n = bin_mask.sum()
        if n == 0:
            return np.nan, np.nan

        # weighted means of coordinates
        cy = (y * bin_mask).sum() / n
        cx = (x * bin_mask).sum() / n

        return cy, cx

    def half_centroids_from_mask(self, mask):
        # TODO: improve this method by splitting only the torso ROI in half
        mask = mask.astype(np.bool)
        if mask.ndim == 3:
            mask = mask[..., 0]
        height, width = mask.shape
        x = np.arange(width)[None, :]  # shape (1, width)
        y = np.arange(height)[:, None]  # shape (height, 1)

        mid = width // 2
        left_mask = mask & (x < mid)
        right_mask = mask & (x >= mid)

        return self.centroid(left_mask, x, y), self.centroid(right_mask, x, y)

    def half_centroids_from_mask_with_markers(self, mask, p1, p2):
        mask = mask.astype(np.bool)
        if mask.ndim == 3:
            mask = mask[..., 0]
        height, width = mask.shape
        x1, y1 = p1
        x2, y2 = p2

        x = np.arange(width)[None, :]  # shape (1, width)
        y = np.arange(height)[:, None]  # shape (height, 1)

        # Signature relative to line P1->P2
        s = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

        left_side = s < 0
        right_side = s >= 0

        left_mask = mask & left_side
        right_mask = mask & right_side

        return self.centroid(left_mask, x, y), self.centroid(right_mask, x, y)

    @staticmethod
    def calculate_center_deviation(dimples_center_gt, dimples_center_pred):
        if np.isnan(dimples_center_gt[0]) and np.isnan(dimples_center_pred[0]):
            return 0, 0

        if np.isnan(dimples_center_gt[0]) or np.isnan(dimples_center_pred[0]):
            return np.nan, np.nan

        return int(dimples_center_gt[0] - dimples_center_pred[0]), int(dimples_center_gt[1] - dimples_center_pred[1])
