"""
Script to run segmentation experiments on a set of target images.

This script selects specific segmenter configurations (e.g., atlas-based and SSL-based),
loads the corresponding segmenters, and runs them on a given directory of images using the ExperimentRunner.

Usage:
    python segment.py

Configuration:
    - TARGET_IMAGES_DIR: Path to the directory containing images to segment.
    - TARGET_IMAGES_DIR_SCHULTER: Alternative path for shoulder images.
    - image_segmenter: List of segmenter instances to run (select by index from segmenter libraries).

To adapt the script for other experiments or datasets, modify the configuration variables at the top of the file.
"""

from segmenter.atlas_segmenter_lib import atlas_segmenter
from segmenter.experiment_runner import ExperimentRunner
from segmenter.ssl_segmenter_lib import ssl_segmenter

# Path to the images to segment
TARGET_IMAGES_DIR = "data/Images/Validation_Data_Small"
TARGET_IMAGES_DIR_SCHULTER = "data/Images/Validation_Data_Schulter"

# Select specific experiments to run
image_segmenter = [
    atlas_segmenter[01],
    ssl_segmenter[01]
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        ExperimentRunner(segmenter, TARGET_IMAGES_DIR).run()
