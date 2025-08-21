import csv
import os
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

        for file_name in tqdm(ground_truths.keys(), desc=f'Validating predictions for {predictions_dir}'):
            if file_name not in predictions.keys():
                print(f"Ground truth {file_name} does not have a corresponding prediction.")
                continue

            dataset = self.parse_dataset(file_name)
            gt = ground_truths[file_name]
            prediction = predictions[file_name]

            metrics = self.compute_metrics(gt, prediction)

            rows.append([
                dataset,
                file_name,
                metrics.dice,
                metrics.precision,
                metrics.recall,
                metrics.n_gt_segments,
                metrics.n_pred_segments
            ])

            dice_by_set.setdefault(dataset, []).append(metrics.dice)
            precision_by_set.setdefault(dataset, []).append(metrics.precision)
            recall_by_set.setdefault(dataset, []).append(metrics.recall)
            n_get_mask_by_set.setdefault(dataset, []).append(metrics.n_gt_segments)
            n_pred_mask_by_set.setdefault(dataset, []).append(metrics.n_pred_segments)

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = os.path.basename(predictions_dir)
        all_csv = output_dir / f"{run_name}_all.csv"
        mean_csv = output_dir / f"{run_name}_mean.csv"

        with open(all_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Dataset', 'File Name', 'Dice', 'Precision', 'Recall', 'Number of GT Segments',
                             'Number of Segmentation Segments'])
            writer.writerows(rows)
        print(f"Validation results saved to {all_csv}")

        # Mean values
        all_dice = []
        all_precision = []
        all_recall = []
        all_n_gt_mask = []
        all_n_seg_mask = []

        for dataset in dice_by_set.keys():
            all_dice.extend(dice_by_set[dataset])
            all_precision.extend(precision_by_set[dataset])
            all_recall.extend(recall_by_set[dataset])
            all_n_gt_mask.extend(n_get_mask_by_set[dataset])
            all_n_seg_mask.extend(n_pred_mask_by_set[dataset])

        mean_all_dice = np.mean(all_dice)
        mean_all_precision = np.mean(all_precision)
        mean_all_recall = np.mean(all_recall)
        mean_all_n_gt_mask = np.mean(all_n_gt_mask)
        mean_all_n_seg_mask = np.mean(all_n_seg_mask)

        with open(mean_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Dataset', 'Mean Dice', 'Mean Precision', 'Mean Recall', 'Mean Number of GT Segments',
                             'Mean Number of Segmentation Segments'])
            for dataset in dice_by_set.keys():
                writer.writerow([dataset, np.mean(dice_by_set[dataset]), np.mean(precision_by_set[dataset]),
                                 np.mean(recall_by_set[dataset]), np.mean(n_get_mask_by_set[dataset]),
                                 np.mean(n_pred_mask_by_set[dataset])])
            writer.writerow(['All Datasets', mean_all_dice, mean_all_precision, mean_all_recall,
                             mean_all_n_gt_mask, mean_all_n_seg_mask])
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

    @staticmethod
    def compute_metrics(gt, pred):
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

        return SegmentMetrics(
            dice=dice,
            precision=precision,
            recall=recall,
            n_gt_segments=n_gt_mask,
            n_pred_segments=n_seg_mask
        )
