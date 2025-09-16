from experiments.ssl_experiments import ssl_experiments
from segmenter.experiment_runner import ExperimentRunner

TARGET_IMAGES_DIR = "data/Images/Validation_Data_Small"

image_segmenter = [
    ssl_experiments[37],
    ssl_experiments[38],
    ssl_experiments[39],
    ssl_experiments[40],
    ssl_experiments[41],
    ssl_experiments[42],
    ssl_experiments[43],
    ssl_experiments[44],
    ssl_experiments[45],
    ssl_experiments[46],
    ssl_experiments[47],
]

if __name__ == "__main__":
    for segmenter in image_segmenter:
        ExperimentRunner(segmenter, TARGET_IMAGES_DIR).run()
