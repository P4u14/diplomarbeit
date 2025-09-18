from experiments.atlas_experiments import atlas_experiments
from segmenter.experiment_runner import ExperimentRunner

TARGET_IMAGES_DIR = "data/Images/Validation_Data_Small"

image_segmenter = [
    atlas_experiments[37],
    atlas_experiments[38],
    atlas_experiments[39],
    atlas_experiments[40],
    atlas_experiments[41],
    atlas_experiments[78],
    atlas_experiments[79],
    atlas_experiments[80],
    atlas_experiments[81],
    atlas_experiments[82],
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        ExperimentRunner(segmenter, TARGET_IMAGES_DIR).run()
