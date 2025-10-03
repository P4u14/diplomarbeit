from abc import ABC, abstractmethod


class IImageSegmenter(ABC):
    """
    Interface for image segmentation workflows.
    All segmenter classes should inherit from this interface and implement the required methods for loading images,
    segmenting them, and saving the segmentation results.
    """

    @abstractmethod
    def load_target_images(self, directory_path):
        """
        Load all target images from a directory for segmentation.

        Args:
            directory_path (str): Path to the directory containing input images.

        Returns:
            list: List of loaded images or image objects for segmentation.
        """
        pass

    @abstractmethod
    def segment_images(self, image_paths):
        """
        Segment a list of images.

        Args:
            image_paths (list): List of image paths or image objects to segment.

        Returns:
            list: List of segmentation results.
        """
        pass

    @abstractmethod
    def save_segmentation(self, segmented_image):
        """
        Save a segmentation result to disk or another output format.

        Args:
            segmented_image: The segmentation result to save.
        """
        pass
