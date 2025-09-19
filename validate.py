"""
Script to validate segmentation experiments.

This script initializes a Validator with a ground truth directory and an output directory.
It iterates over a predefined list of atlas segmentation experiments, validates each experiment's
predictions directory against the ground truth data, and saves the validation metrics.
"""

import os

from validation.metrics.gt_to_diers_dimples.gt_center_angle_diers_error_metric import GTCenterAngleDiersErrorMetric
from validation.metrics.gt_to_diers_dimples.gt_center_angle_diers_success_metric import GTCenterAngleDiersSuccessMetric
from validation.metrics.gt_to_diers_dimples.gt_center_diers_error_left_metric import GTCenterDiersErrorLeftMetric
from validation.metrics.gt_to_diers_dimples.gt_center_diers_error_right_metric import GTCenterDiersErrorRightMetric
from validation.metrics.gt_to_diers_dimples.gt_center_diers_success_metric import GTCenterDiersSuccessMetric
from validation.metrics.pred_to_diers_dimples.segment_center_diers_success_metric import \
    SegmentsCenterDiersSuccessMetric
from validation.metrics.pred_to_diers_dimples.segments_center_angle_diers_error_metric import \
    SegmentsCenterAngleDiersErrorMetric
from validation.metrics.pred_to_diers_dimples.segments_center_angle_diers_success_metric import \
    SegmentsCenterAngleDiersSuccessMetric
from validation.metrics.pred_to_diers_dimples.segments_center_diers_error_left_metric import \
    SegmentsCenterDiersErrorLeftMetric
from validation.metrics.pred_to_diers_dimples.segments_center_diers_error_right_metric import \
    SegmentsCenterDiersErrorRightMetric
from validation.metrics.segment_centers.segments_center_angle_error_metric import SegmentsCenterAngleErrorMetric
from validation.metrics.segment_centers.segments_center_angle_success_metric import SegmentsCenterAngleSuccessMetric
from validation.metrics.segment_centers.segments_center_error_left_metric import SegmentsCenterErrorLeftMetric
from validation.metrics.segment_centers.segments_center_error_metric import SegmentsCenterErrorMetric
from validation.metrics.segment_centers.segments_center_error_right_metric import SegmentsCenterErrorRightMetric
from validation.metrics.segment_centers.segments_center_success_metric import SegmentsCenterSuccessMetric
from validation.metrics.standard.dice_metric import DiceMetric
from validation.metrics.standard.n_segments_ratio_metric import NSegmentsRatioMetric
from validation.metrics.standard.precision_metric import PrecisionMetric
from validation.metrics.standard.recall_metric import RecallMetric
from validation.validator import Validator

# Path to the base directory containing segmentation result subdirectories
BASE_PATH = 'data/Results/Segmentation_Results/Atlas'

# List of experiment names to validate
EXPERIMENTS_TO_VALIDATE = [
    'Atlas_Experiment01',
]

# List of metric instances to compute
METRICS = [
    DiceMetric(), PrecisionMetric(), RecallMetric(), NSegmentsRatioMetric(),
    SegmentsCenterErrorMetric(),
    SegmentsCenterErrorLeftMetric(), SegmentsCenterErrorRightMetric(), SegmentsCenterSuccessMetric(),
    SegmentsCenterAngleErrorMetric(), SegmentsCenterAngleSuccessMetric(),
    SegmentsCenterDiersErrorLeftMetric(), SegmentsCenterDiersErrorRightMetric(), SegmentsCenterDiersSuccessMetric(),
    SegmentsCenterAngleDiersErrorMetric(), SegmentsCenterAngleDiersSuccessMetric(),
    GTCenterDiersErrorLeftMetric(), GTCenterDiersErrorRightMetric(), GTCenterDiersSuccessMetric(),
    GTCenterAngleDiersErrorMetric(), GTCenterAngleDiersSuccessMetric(),
]

# Directory where validation output files will be saved
OUTPUT_DIR = "data/Results/Validation_old/"

if __name__ == "__main__":
    validator = Validator(ground_truth_dir="data/Images/Validation_Data_Small", output_dir=OUTPUT_DIR, metrics=METRICS)

    for experiment in EXPERIMENTS_TO_VALIDATE:
        validator.validate(predictions_dir=os.path.join(BASE_PATH, experiment))
