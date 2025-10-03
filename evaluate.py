"""
Script to validate segmentation experiments.

This script initializes an Evaluator with a ground truth directory and an output directory.
It iterates over a predefined list of segmentation experiments, validates each experiment's
predictions directory against the ground truth data, and saves the validation metrics.

Usage:
    python evaluate.py

Configuration:
    - BASE_PATH: Path to the base directory containing segmentation result subdirectories.
    - EXPERIMENTS_TO_VALIDATE: List of experiment names to validate.
    - METRICS: List of metric instances to compute for each experiment.
    - OUTPUT_DIR: Directory where validation output files will be saved.

The script can be adapted to validate different sets of experiments or use different ground truth directories
by modifying the configuration variables at the top of the file.
"""

import os

from evaluation.evaluator import Evaluator
from evaluation.metrics.gt_to_diers_dimples.gt_center_angle_diers_error_metric import GTCenterAngleDiersErrorMetric
from evaluation.metrics.gt_to_diers_dimples.gt_center_angle_diers_success_metric import GTCenterAngleDiersSuccessMetric
from evaluation.metrics.gt_to_diers_dimples.gt_center_diers_error_left_metric import GTCenterDiersErrorLeftMetric
from evaluation.metrics.gt_to_diers_dimples.gt_center_diers_error_right_metric import GTCenterDiersErrorRightMetric
from evaluation.metrics.gt_to_diers_dimples.gt_center_diers_success_metric import GTCenterDiersSuccessMetric
from evaluation.metrics.pred_to_diers_dimples.segment_center_diers_success_metric import \
    SegmentsCenterDiersSuccessMetric
from evaluation.metrics.pred_to_diers_dimples.segments_center_angle_diers_error_metric import \
    SegmentsCenterAngleDiersErrorMetric
from evaluation.metrics.pred_to_diers_dimples.segments_center_angle_diers_success_metric import \
    SegmentsCenterAngleDiersSuccessMetric
from evaluation.metrics.pred_to_diers_dimples.segments_center_diers_error_left_metric import \
    SegmentsCenterDiersErrorLeftMetric
from evaluation.metrics.pred_to_diers_dimples.segments_center_diers_error_right_metric import \
    SegmentsCenterDiersErrorRightMetric
from evaluation.metrics.segment_centers.segments_center_angle_error_metric import SegmentsCenterAngleErrorMetric
from evaluation.metrics.segment_centers.segments_center_angle_success_metric import SegmentsCenterAngleSuccessMetric
from evaluation.metrics.segment_centers.segments_center_error_left_metric import SegmentsCenterErrorLeftMetric
from evaluation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric
from evaluation.metrics.segment_centers.segments_center_error_right_metric import SegmentsCenterErrorRightMetric
from evaluation.metrics.segment_centers.segments_center_success_metric import SegmentsCenterSuccessMetric
from evaluation.metrics.standard.dice_metric import DiceMetric
from evaluation.metrics.standard.n_segments_gt_metric import NSegmentsGTMetric
from evaluation.metrics.standard.n_segments_pred_metrics import NSegmentsPredMetric
from evaluation.metrics.standard.n_segments_ratio_metric import NSegmentsRatioMetric
from evaluation.metrics.standard.precision_metric import PrecisionMetric
from evaluation.metrics.standard.recall_metric import RecallMetric

# Path to the base directory containing segmentation result subdirectories
BASE_PATH = 'data/Results/Segmentation_Results'

# List of experiment names to validate
EXPERIMENTS_TO_VALIDATE = [
    'Atlas/Atlas_Experiment01',
    'SSL/SSL_Experiment01',
]

# List of metric instances to compute
METRICS = [
    DiceMetric(), PrecisionMetric(), RecallMetric(),
    NSegmentsGTMetric(), NSegmentsPredMetric(), NSegmentsRatioMetric(),
    SegmentsCenterErrorMetric(),
    SegmentsCenterErrorLeftMetric(), SegmentsCenterErrorRightMetric(), SegmentsCenterSuccessMetric(),
    SegmentsCenterAngleErrorMetric(), SegmentsCenterAngleSuccessMetric(),
    SegmentsCenterDiersErrorLeftMetric(), SegmentsCenterDiersErrorRightMetric(), SegmentsCenterDiersSuccessMetric(),
    SegmentsCenterAngleDiersErrorMetric(), SegmentsCenterAngleDiersSuccessMetric(),
    GTCenterDiersErrorLeftMetric(), GTCenterDiersErrorRightMetric(), GTCenterDiersSuccessMetric(),
    GTCenterAngleDiersErrorMetric(), GTCenterAngleDiersSuccessMetric(),
]

# Directory where validation output files will be saved
OUTPUT_DIR = "data/Results/Validation/"

if __name__ == "__main__":
    evaluator = Evaluator(ground_truth_dir="data/Images/Validation_Data_Small", output_dir=OUTPUT_DIR, metrics=METRICS)
    # evaluator = Evaluator(ground_truth_dir="data/Images/Validation_Data_Schulter", output_dir=OUTPUT_DIR, metrics=METRICS)

    for experiment in EXPERIMENTS_TO_VALIDATE:
        evaluator.evaluate(predictions_dir=os.path.join(BASE_PATH, experiment))
