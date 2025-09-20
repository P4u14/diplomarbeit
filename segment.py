from segmenter.atlas_segmenter_lib import atlas_segmenter
from segmenter.experiment_runner import ExperimentRunner

# Path to the images to segment
TARGET_IMAGES_DIR = "data/Images/Validation_Data_Small"

# Select specific experiments to run
image_segmenter = [
    atlas_segmenter[37],
    atlas_segmenter[38],
    atlas_segmenter[39],
    atlas_segmenter[40],
    atlas_segmenter[41],
    atlas_segmenter[78],
    atlas_segmenter[79],
    atlas_segmenter[80],
    atlas_segmenter[81],
    atlas_segmenter[82],
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        ExperimentRunner(segmenter, TARGET_IMAGES_DIR).run()
