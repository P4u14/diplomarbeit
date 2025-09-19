import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from tqdm import tqdm


class Validator:

    def __init__(self, ground_truth_dir, output_dir, metrics):
        """
        Initialize Validator.
        Parameters:
            ground_truth_dir (str): Path to ground truth masks.
            output_dir (str): Directory to save CSV results.
            metrics (list[Metric], optional): List of metric instances to compute. If None, auto-discover.
        """
        self.ground_truth_dir = ground_truth_dir
        self.ground_truths = self.load_masks(ground_truth_dir)
        self.output_dir = output_dir
        self.vp_dm_distances = self.load_vp_dm_distances()
        self.metrics = metrics
        self.health_status_dict = self.load_health_status_dict()


    def validate(self, predictions_dir):
        # Load ground truth and prediction masks
        ground_truths = self.ground_truths
        predictions = self.load_masks(predictions_dir)

        # Prepare data structures for computed_metric_results
        metric_scores = []
        metric_scores_by_dataset = defaultdict(list)
        metric_scores_by_health_status = defaultdict(list)

        # Iterate over all ground truth files and validate predictions
        for file_name in tqdm(ground_truths.keys(), desc=f'Validating predictions for {predictions_dir}'):
            if file_name not in predictions.keys():
                print(f"Ground truth {file_name} does not have a corresponding prediction.")
                continue

            # Load ground truth and prediction masks for the current image
            dataset = self.parse_dataset(file_name)
            gt = ground_truths[file_name] > 0  # binary mask
            pred = predictions[file_name] > 0  # binary mask

            # Compute TP, FP, FN as base metrics
            computed_metric_results = {
                'TP': np.logical_and(gt, pred).sum(),
                'FP': np.logical_and(~gt, pred).sum(),
                'FN': np.logical_and(gt, ~pred).sum()
            }

            # Compute image metadata
            image_metadata = self.load_image_metadata(file_name)

            # Compute metrics sequentially, allowing composite metrics to use previous computed_metric_results
            for m in self.metrics:
                computed_metric_results[m.name] = m.compute(gt, pred, computed_metric_results, image_metadata)
            # Collect metric values (keep infinities for per-image output)
            values = [computed_metric_results[m.name] for m in self.metrics]

            # Determine health status for the patient from the patient index
            pat_idx = self.parse_patient_index_from_image_path(file_name)
            sick = self.health_status_dict.get(pat_idx)

            # Add metric scores for this file to the overall list
            metric_scores.append([dataset, file_name, sick] + values)

            # Add metric scores for this file to the dataset-specific list
            metric_scores_by_dataset[dataset].append(values)

            # Add metric scores for this file to the health status-specific list
            if sick == 1.0:
                metric_scores_by_health_status['Sick'].append(values)
            elif sick == 0.0:
                metric_scores_by_health_status['Healthy'].append(values)

        # prepare output directories
        all_csv, mean_csv = self.create_output_files(predictions_dir)
        # write all per-file metric_scores
        self.save_per_image_metrics(all_csv, metric_scores)
        # Read duration metric_scores
        avg_image_duration, total_duration = self.read_segmentation_duration(predictions_dir)
        # Write mean metric_scores grouped by dataset, by sick status and overall
        self.save_mean_metrics(avg_image_duration, mean_csv, metric_scores_by_dataset, metric_scores_by_health_status,
                               total_duration)




    def load_image_metadata(self, file_name):
        # Load markers
        vp, dm, dl_diers, dr_diers = self.load_markers(file_name)

        # Calc pixel ratio in mm
        pat_idx = self.parse_patient_index_from_image_path(file_name)
        pixel_size = self.compute_distance_per_pixel(pat_idx, vp, dm)

        return {
            'vp': vp,
            'dm': dm,
            'dl_diers': dl_diers,
            'dr_diers': dr_diers,
            'pixel_size_mm': pixel_size
        }


    def load_markers(self, file_name, markers_file="data/Info_sheets/Markerpositionen.csv"):
        # Check if markers are available for this image
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

        # Check if markers are available for this patient
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
                    vp = (int(row['X_VP'] / 10), int(row['Y_VP']) / 10)
                    dm = (int(row['X_DM'] / 10), int(row['Y_DM']) / 10)
                    dl_diers = (int(row['X_DL']) / 10, int(row['Y_DL']) / 10)
                    dr_diers = (int(row['X_DR']) / 10, int(row['Y_DR']) / 10)
                    return vp, dm, dl_diers, dr_diers

        # If no markers found, return None
        return None, None, None, None


    def save_mean_metrics(self, avg_image_duration, mean_csv, metric_scores_by_dataset, metric_scores_by_health_status,
                          total_duration):
        with open(mean_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header_mean = ['Dataset'] + [m.name for m in self.metrics] + ['Total duration',
                                                                          'Average duration per image']
            writer.writerow(header_mean)
            # per-dataset means
            for ds, lst in metric_scores_by_dataset.items():
                means = [_safe_nanmean([row[i] for row in lst]) for i in range(len(self.metrics))]
                writer.writerow([ds] + means + ['', ''])
            # by sick status
            for label, lst in metric_scores_by_health_status.items():
                means = [_safe_nanmean([row[i] for row in lst]) for i in range(len(self.metrics))]
                writer.writerow([label] + means + ['', ''])
            # overall
            all_vals = [v for lst in metric_scores_by_dataset.values() for v in lst]
            means = [_safe_nanmean([row[i] for row in all_vals]) for i in range(len(self.metrics))]
            writer.writerow(['All Datasets'] + means + [total_duration, avg_image_duration])
        print(f"Mean validation results saved to {mean_csv}")


    @staticmethod
    def read_segmentation_duration(predictions_dir):
        _duration_file = Path(predictions_dir) / 'duration.txt'
        if _duration_file.exists():
            with open(_duration_file, 'r') as _df:
                _lines = _df.readlines()
            if len(_lines) >= 2:
                total_duration = _lines[0].split(':', 1)[1].strip()
                avg_image_duration = _lines[1].split(':', 1)[1].strip()
            else:
                total_duration = ''
                avg_image_duration = ''
        else:
            total_duration = ''
            avg_image_duration = ''
        return avg_image_duration, total_duration


    def save_per_image_metrics(self, all_csv, metric_scores):
        with open(all_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Dataset', 'File Name', 'Sick'] + [m.name for m in self.metrics]
            writer.writerow(header)
            writer.writerows(metric_scores)
        print(f"Validation results saved to {all_csv}")


    def create_output_files(self, predictions_dir):
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = os.path.basename(predictions_dir)
        all_csv = output_dir / f"{run_name}_all.csv"
        mean_csv = output_dir / f"{run_name}_mean.csv"
        return all_csv, mean_csv


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


    def compute_distance_per_pixel(self, patient_idx, vp, dm):
        """
        Compute millimeters per pixel using known physical distance between VP and DM markers.
        """
        if vp is None or dm is None:
            return None
        pixel_dist = np.hypot(dm[0] - vp[0], dm[1] - vp[1])
        if pixel_dist == 0:
            return None
        mm_dist = self.vp_dm_distances.get(patient_idx)
        if mm_dist is None:
            return None
        return mm_dist / pixel_dist


    @staticmethod
    def load_health_status_dict(file_path="data/Info_Sheets/All_Data_Renamed_overview.csv"):
        """Load the mapping from Patientenindex to Krank value."""
        health_status_dict = {}
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pat_idx = row.get('Patientenindex')
                sick = row.get('Krank')
                if pat_idx and sick:
                    health_status_dict[pat_idx] = float(sick)
        return health_status_dict


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
    # filter out None, NaN, and infinite values
    clean = [v for v in arr if v is not None and np.isfinite(v)]
    if len(clean) == 0:
        return None
    return float(np.mean(clean))
