from experiments.atlas_experiments import atlas_experiments
from segmenter.experiment_runner import ExperimentRunner

TARGET_IMAGES_DIR = "data/Images/Validation_Data_Small"

image_segmenter = [
    # atlas_experiments[4],
    atlas_experiments[5],
    atlas_experiments[6],
    atlas_experiments[10],
    atlas_experiments[11],
    atlas_experiments[12],
    atlas_experiments[25],
    atlas_experiments[26],
    atlas_experiments[27],
    atlas_experiments[28],
    atlas_experiments[29],
    atlas_experiments[30],
    atlas_experiments[31],
    atlas_experiments[32],
    atlas_experiments[33],
    atlas_experiments[34],
    atlas_experiments[35],
    atlas_experiments[36],
    atlas_experiments[45],
    atlas_experiments[46],
    atlas_experiments[47],
    atlas_experiments[51],
    atlas_experiments[52],
    atlas_experiments[53],
    atlas_experiments[66],
    atlas_experiments[67],
    atlas_experiments[68],
    atlas_experiments[69],
    atlas_experiments[70],
    atlas_experiments[71],
    atlas_experiments[72],
    atlas_experiments[73],
    atlas_experiments[74],
    atlas_experiments[75],
    atlas_experiments[76],
    atlas_experiments[77],
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        ExperimentRunner(segmenter, TARGET_IMAGES_DIR).run()
