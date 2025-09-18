from experiments.atlas_experiments import atlas_experiments
from segmenter.experiment_runner import ExperimentRunner

TARGET_IMAGES_DIR = "data/Images/Validation_Data_Small"

image_segmenter = [
    atlas_experiments[33],
    atlas_experiments[74],
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        ExperimentRunner(segmenter, TARGET_IMAGES_DIR).run()
