import os
from abc import ABC

import numpy as np
from skimage import io

from segmenter.image_segmenter import IImageSegmenter
from target_image.target_image import TargetImage


class BaseSegmenter(IImageSegmenter, ABC):
    """
    Abstract base class for image segmentation workflows.
    Provides common logic for loading target images, saving segmentations, and managing preprocessing and refinement steps.
    Subclasses must implement the segment_images method.

    Args:
        output_dir (str): Directory to save segmentation results.
        preprocessing_steps (list): List of preprocessing step instances to apply to images.
        segmentation_refiner (object): Optional refinement step for segmentation results.
        img_extension (str): File extension for input images (default: '.png').
    """
    def __init__(self, output_dir, preprocessing_steps, segmentation_refiner, img_extension=".png"):
        """
        Initialize the BaseSegmenter.
        Args:
            output_dir (str): Directory to save segmentation results.
            preprocessing_steps (list): List of preprocessing step instances to apply to images.
            segmentation_refiner (object): Optional refinement step for segmentation results.
            img_extension (str): File extension for input images (default: '.png').
        """
        self.output_dir = output_dir
        self.preprocessing_steps = preprocessing_steps
        self.segmentation_refiner = segmentation_refiner
        self.img_extension = img_extension

    def load_target_images(self, directory_path):
        """
        Load all target images from a directory, excluding mask files.
        Args:
            directory_path (str): Path to the directory containing input images.
        Returns:
            list[TargetImage]: List of TargetImage objects for segmentation.
        """
        target_images = []
        for file in os.listdir(directory_path):
            if file.endswith(self.img_extension) and "-mask" not in file:
                target_images.append(TargetImage(os.path.join(directory_path, file)))
        return target_images

    def segment_images(self, target_images):
        """
        Abstract method for segmenting a list of target images.
        Must be implemented by subclasses.
        Args:
            target_images (list[TargetImage]): List of TargetImage objects to segment.
        Raises:
            NotImplementedError: Always, unless implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement segment method")

    def save_segmentation(self, target_segmentation):
        """
        Save a segmentation mask to the output directory as a PNG file.
        Args:
            target_segmentation (TargetSegmentation): Segmentation result to save. Must have 'output_path' and 'result_mask'.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filepath = os.path.join(self.output_dir, target_segmentation.output_path)
        segmentation = target_segmentation.result_mask
        io.imsave(str(filepath), (segmentation * 255).astype(np.uint8))
        print("Saved segmentation mask to {}".format(filepath))
