import os

from skimage import io


class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = io.imread(self.image_path)
        self.preprocessed_image = None
        self.preprocessing_steps = []


class TargetSegmentation:
    def __init__(self, output_path, result_mask):
        self.output_path = output_path
        self.result_mask = result_mask