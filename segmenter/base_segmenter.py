import os
from abc import ABC

import numpy as np
from skimage import io

from segmenter.image_segmenter import IImageSegmenter
from target_image.target_image import TargetImage


class BaseSegmenter(IImageSegmenter, ABC):

    def __init__(self, output_dir, preprocessing_steps, segmentation_refiner, img_extension=".png"):
        self.output_dir = output_dir
        self.preprocessing_steps = preprocessing_steps
        self.segmentation_refiner = segmentation_refiner
        self.img_extension = img_extension


    def load_target_images(self, directory_path):
        target_images = []
        for file in os.listdir(directory_path):
            if file.endswith(self.img_extension) and "-mask" not in file:
                target_images.append(TargetImage(os.path.join(directory_path, file)))
        return target_images

    def segment(self, target_images):
        raise NotImplementedError("Subclasses must implement segment method")

    def save_segmentation(self, target_segmentation):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filepath = os.path.join(self.output_dir, target_segmentation.output_path)
        segmentation = target_segmentation.result_mask
        io.imsave(str(filepath), (segmentation * 255).astype(np.uint8))
        print("Saved segmentation mask to {}".format(filepath))
