from segmenter.atlas_segmenter_lib import atlas_segmenter
from segmenter.experiment_runner import ExperimentRunner
from segmenter.ssl_segmenter_lib import ssl_segmenter

# Path to the images to segment
TARGET_IMAGES_DIR = "data/Images/Validation_Data_Small"
TARGET_IMAGES_DIR_SCHULTER = "data/Images/Validation_Data_Schulter"

# Select specific experiments to run
image_segmenter = [
    # atlas_segmenter[83]
    # ssl_segmenter[47],
    # ssl_segmenter[48],
    # ssl_segmenter[95],
    ssl_segmenter[96]
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        ExperimentRunner(segmenter, TARGET_IMAGES_DIR_SCHULTER).run()
